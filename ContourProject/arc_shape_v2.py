# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator
#
# MIT License
#
# Copyright (c) 2018-2024
#     Ivo Steinbrecher
#     Institute for Mathematics and Computer-Based Simulation
#     Universitaet der Bundeswehr Muenchen
#     https://www.unibw.de/imcs-en
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""
This script contains a tutorial for meshpy. Most basic functionality is covered
by this tutorial. For more information have a closer look at the test cases,
as they cover all functionality.
"""

# Import python modules.
import numpy as np
import autograd.numpy as npAD
import os

# Import the objects we need from meshpy.
from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.core.mesh import Mesh
from meshpy.core.rotation import Rotation
from meshpy.four_c.boundary_condition import BoundaryCondition
from meshpy.four_c.element_beam import Beam3rHerm2Line3
from meshpy.four_c.function import Function
from meshpy.four_c.input_file import InputFile
from meshpy.four_c.material import MaterialReissner
from meshpy.four_c.run_four_c import run_four_c
from meshpy.utils.nodes import get_single_node, find_close_nodes
from meshpy.mesh_creation_functions.beam_basic_geometry import (
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_line,
)
from meshpy.mesh_creation_functions.beam_curve import create_beam_mesh_curve


def create_beams_wrapped_around_cylinder(base_dir, preview=False):

    interval = [0, 6] # t
    cylinder_radius = 3.25 # radius of wrapping cylinder
    n_intersections = 2 # min 2
    number_of_beams = 16 # each direction normally 72 in total
    compression_factor = 0.95 # Maximum compression. Probably should be around 0.9-0.95
    compressed_part = 0.1 # ratio how much of each beam is compressed from start
    beam_radius = 0.03 # radius of the beam
    youngs_modulus = 30000 # N/mm^2 Young's modulus of the beam material
    # penalty parameters 500 and 50 solved converging problem
    positional_coupling_penalty = 500 # penalty for positional coupling
    rotational_coupling_penalty = 50 # penalty for rotational coupling
    # radius of marker max 0.25 mm

    n_el = 4*(n_intersections-1)*number_of_beams
    time_step = 0.1
    num_steps = 10

    
    mesh = Mesh()
    mat = MaterialReissner(youngs_modulus=youngs_modulus, radius=beam_radius)
    beam_object = Beam3rHerm2Line3

    # use find_close_nodes() instead of this func

    def find_intersections_and_apply_coupling(mesh, beams, positional_coupling_penalty, rotational_coupling_penalty):
        """
        Find intersections between beams and apply coupling.
        """
        # Get all nodes from all beams
        #all_nodes = []
        #for beam in beams:
        #    all_nodes.extend(beam["line"].get_all_nodes())
        
        # Use find_close_nodes to find intersecting nodes
        # Returns list of lists where each inner list contains nodes that are close to each other
        close_node_groups = find_close_nodes(mesh.nodes)
        print("\nFound close node groups:")
        for group_idx, node_group in enumerate(close_node_groups):
            if len(node_group) > 1:  # Only print groups with multiple nodes
                print(f"\nGroup {group_idx + 1}:")
                for node_idx, node in enumerate(node_group):
                    coords = node.coordinates
                    print(f"Node {node_idx + 1}: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})")
                print("-" * 50)
        # Create GeometrySet for intersecting nodes
        # intersecting nodes with first node as initial geometry
        first_group = next((group for group in close_node_groups if len(group) > 1), None)
        if not first_group:
            return None
            
        intersecting_nodes = GeometrySet(first_group[0])
        
        # Apply coupling for each group of close nodes
        for node_group in close_node_groups:
            #print("\nLevel1:")
            if len(node_group) > 1:  # Only process groups with multiple nodes
                #print("\nLevel2:")
                # Add nodes to geometry set
                for node in node_group:
                    intersecting_nodes.add(node)
                
                # Apply coupling between nodes in this group
                
                mesh.couple_nodes(
                    nodes=[node_group[0], node_group[1]],
                    coupling_type=mpy.bc.point_coupling_penalty,
                    coupling_dof_type=f"POSITIONAL_PENALTY_PARAMETER {positional_coupling_penalty} ROTATIONAL_PENALTY_PARAMETER {rotational_coupling_penalty}"
                )
                
        
        return intersecting_nodes

    def calculate_displacement_for_cylinder(coordinates, new_radius):
        """
        Calculates displacement for cylinder transformation.
        """
        x = coordinates[0]
        y = coordinates[1]
        # Calculate the original radius
        original_radius = np.sqrt(x**2 + y**2)
        # Calculate the scaling factor
        scaling_factor = new_radius / original_radius
        # Scale the x and y coordinates
        new_x = x * scaling_factor
        new_y = y * scaling_factor
        # return the displacement
        return np.array([new_x-x, new_y-y, 0])
    
    def calculate_angle_for_intersections(n_intersections, interval, radius):
        """
        Calculate required angle for desired number of intersections
        
        Args:
            n_intersections: desired number of intersections
            interval: [start, end] of beam
            radius: cylinder radius
        """
        length = interval[1] - interval[0]
        # Adjust for the interval offset
        tan_yz = ((n_intersections - 1) * np.pi * radius) / length
        alpha_degrees = np.degrees(np.arctan(tan_yz))
        return tan_yz, alpha_degrees

    def create_multiple_beams_shifted_in_y(
        mesh,
        number_of_beams,
        cylinder_radius,
        interval,
        n_el,
        add_sets,
        material,
        beam_object,
        tan_yz,
        compression_factor,
        compressed_part
    ):
        """
        Creates multiple beams, each shifted in y by an equal amount of 2π/number_of_beams.
        """
        """
        def n_shape_yz_complete(t):
            x = cylinder_radius
            R = interval[1]/2  # Radius of the arc (half of total interval)
            arc_height = R  # Maximum height of the arc
            
            # Parameter from 0 to π (half circle)
            theta = (t/(interval[1])) * npAD.pi
            
            # Calculate coordinates
            z = R * npAD.sin(theta)  # z goes up and down
            y = tan_yz * R * (1 - npAD.cos(theta))  # y increases throughout
            
            return npAD.array([x, y, z])
        """

        beams = []
        beams_start = []
        beams_end = []
        
        # Calculate arc radius based on the interval and desired shape
        arc_length = interval[1] - interval[0]
        R = arc_length / 2  # Radius matches what was previously used in n_shape_yz_complete

        # Create straight line shape function (along y-axis)
        def straight_line(t):
            x = cylinder_radius
            # Calculate y coordinate that matches the final y-position of previous arc
            y = tan_yz * R * (t / interval[1])  # Linear increase in y
            z = 0  # Flat line initially (will be transformed to arc)
            return npAD.array([x, y, z])

        for i in range(number_of_beams):
            shift_i = (2.0 * npAD.pi * cylinder_radius / number_of_beams) * i
            
            def shape_with_shift(t, shift=shift_i):
                base = straight_line(t)
                return npAD.array([base[0], base[1] + shift, base[2]])

            dir1 = create_beam_mesh_curve(
                mesh,
                beam_object,
                material,
                shape_with_shift,
                interval=interval,
                n_el=n_el,
                add_sets=add_sets
            )
            # 

            beams.append(dir1)
            beams_start.append(dir1["start"])
            beams_end.append(dir1["end"])

            # Get the submesh representing just this beam
            beam_nodes = dir1["line"].get_all_nodes()
            
            # Create a temporary submesh to apply the arc transformation
            """
            temp_mesh = Mesh()
            for node in beam_nodes:
                temp_mesh.nodes.append(node)
            
            # Transform the straight line to an arc
            # The radius must be greater than half the line length
            arc_radius = R / np.sin(np.pi/2)  # Calculate required radius
            temp_mesh.transform_line_to_arc(radius=arc_radius, fixed_endpoints=True)
            """

        #mesh.display_pyvista()
        #mesh.wrap_around_cylinder(radius=cylinder_radius)
        #find_intersections_and_apply_coupling(mesh, beams, positional_coupling_penalty, rotational_coupling_penalty) # change this
        mpy.check_overlapping_elements = False

        print("\nBeam start and end point coordinates:")
        for i, beam in enumerate(beams):
            start_node = beam["start"].get_all_nodes()[0]  # Get the first node from the geometry set
            start_coords = start_node.coordinates
            print(f"Beam {i + 1}: ({start_coords[0]:.3f}, {start_coords[1]:.3f}, {start_coords[2]:.3f})")
            end_node = beam["end"].get_all_nodes()[0]  # Get the first node from the geometry set
            end_coords = end_node.coordinates
            print(f"Beam {i + 1}: ({end_coords[0]:.3f}, {end_coords[1]:.3f}, {end_coords[2]:.3f})")

        new_radius = cylinder_radius * (1.0 - compression_factor)

        # write a for loop for all nodes in mesh like below
        
        for node in mesh.nodes:
            if not node.is_middle_node:
                #node in beams_start: This is wrong I dont know why
                
                if  np.linalg.norm(node.coordinates[2] - interval[0]) < 1e-9:                #node in beams_start:
                    print(f"Start node coordinates: {node.coordinates}")
                    # boundary condition with radial and axial=0 displacement
                    node_set = GeometrySet(node)
                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                "NUMDOF 9 ONOFF 0 0 1 0 0 0 0 0 0 "
                                "VAL 0 0 0 0 0 0 0 0 0 "
                                "FUNCT 0 0 0 0 0 0 0 0 0"
                            ),
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                    
                    """
                    displacement = calculate_displacement_for_cylinder(
                            node.coordinates, 
                            new_radius
                        )
                    displacement_x = Function(
                        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                        "VARIABLE 0 NAME a TYPE linearinterpolation "
                        "NUMPOINTS 3 TIMES 0.0 1.0 1000.0 VALUES 0.0 {} {}".format(
                            displacement[0],  # x-component of displacement
                            displacement[0]
                        )
                    )
                    displacement_y = Function(
                        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                        "VARIABLE 0 NAME a TYPE linearinterpolation "
                        "NUMPOINTS 3 TIMES 0.0 1.0 1000.0 VALUES 0.0 {} {}".format(
                            displacement[1],  # y-component of displacement
                            displacement[1]
                        )
                    )
                    mesh.add(displacement_x)
                    mesh.add(displacement_y)

                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                "NUMDOF 9 ONOFF 1 1 1 0 0 0 0 0 0 "  # Fix z also only x and y translations
                                "VAL 1 1 0 0 0 0 0 0 0 "
                                "FUNCT {} {} 0 0 0 0 0 0 0"  # Use displacement functions for x,y
                            ),
                            format_replacement=[displacement_x, displacement_y],  # Use the displacement functions
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                    
                """
                elif node.coordinates[2] < ((interval[1] + interval[0])/2)*compressed_part:  # z < 5 for interval [0, 10]
                    # add here the boundary condition with radial displacement according to coordinate
                    node_set = GeometrySet(node)

                    displacement = calculate_displacement_for_cylinder(
                            node.coordinates, 
                            new_radius
                        )
                    #print(f"Node coordinates: {node.coordinates}")
                    #print(f"Calculated displacement: {displacement}")

                    # Create displacement functions with time interpolation
                    displacement_x = Function(
                        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                        "VARIABLE 0 NAME a TYPE linearinterpolation "
                        "NUMPOINTS 3 TIMES 0.0 1.0 1000.0 VALUES 0.0 {} {}".format(
                            displacement[0],  # x-component of displacement
                            displacement[0]
                        )
                    )
                    displacement_y = Function(
                        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                        "VARIABLE 0 NAME a TYPE linearinterpolation "
                        "NUMPOINTS 3 TIMES 0.0 1.0 1000.0 VALUES 0.0 {} {}".format(
                            displacement[1],  # y-component of displacement
                            displacement[1]
                        )
                    )
                    mesh.add(displacement_x)
                    mesh.add(displacement_y)

                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                "NUMDOF 9 ONOFF 1 1 0 0 0 0 0 0 0 "  # Fix only x and y translations
                                "VAL 1 1 0 0 0 0 0 0 0 "
                                "FUNCT {} {} 0 0 0 0 0 0 0"  # Use displacement functions for x,y
                            ),
                            format_replacement=[displacement_x, displacement_y],  # Use the displacement functions
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                    #node in beams_end: This didnot work I dont know why
                """
                elif np.linalg.norm(node.coordinates[2] - interval[0])<1e-9: 
                    # add here find end nodes
                    print(f"End node coordinates: {node.coordinates}")
                    node_set = GeometrySet(node)
                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                "NUMDOF 9 ONOFF 0 0 0 0 0 0 0 0 0 "  # Fix only x and y translations
                                # kola bardağı gibi oluyor x ve y sınırlayınca
                                "VAL 0 0 0 0 0 0 0 0 0 "
                                "FUNCT 0 0 0 0 0 0 0 0 0"  # Use displacement functions for x,y
                            ),
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                    """
                    #pass
    
            
    tan_yz, degrees = calculate_angle_for_intersections(n_intersections, interval, cylinder_radius)

    print(f"degrees: {degrees}")

    

    create_multiple_beams_shifted_in_y(
        mesh,
        number_of_beams,
        cylinder_radius,
        interval,
        n_el,
        True,
        mat,
        beam_object,
        tan_yz,
        compression_factor,
        compressed_part
    )

    
    # The vtk output will also show all node sets for BCs on the mesh.
    mesh.write_vtk("simple_beam", base_dir)

    # The object InputFile is a mesh, but can also store 4C input parameters.
    # Additionally we load an existing solid mesh. This shows how solid, or in
    # general, volume elements (fluid, ...) can be combined with beam elements.
    # Everything from the volume input file will be included in the combined
    # input file, e.g. BC, loads, materials, solver parameters, ... .
    solid_dat_path = os.path.join(
        os.path.dirname(__file__), "Input.dat"
    )
    input_file = InputFile()

    # Add the beam geometry to the input file.
    input_file.add(mesh)

    # Add the input parameters.
    input_file.add(
        f"""
        ------------------------------------------------------------------TITLE
        meshpy tutorial
        -----------------------------------------------------------PROBLEM TYPE
        PROBLEMTYPE                           Structure
        RESTART                               0
        ---------------------------------------------------------------------IO
        OUTPUT_BIN                            no
        STRUCT_DISP                           yes
        FILESTEPS                             1000
        VERBOSITY                             Standard
        STRUCT_STRAIN                         yes
        STRUCT_STRESS                         yes
        -----------------------------------------------------STRUCTURAL DYNAMIC
        LINEAR_SOLVER                         1
        INT_STRATEGY                          Standard
        DYNAMICTYPE                           Statics
        RESULTSEVRY                           1
        NLNSOL                                fullnewton
        TIMESTEP                              {time_step}
        NUMSTEP                               {num_steps}
        MAXTIME                               2.0
        ---------------------------------------------------------------SOLVER 1
        NAME                                  Structure_Solver
        SOLVER                                Superlu
        --------------------------------------------------IO/RUNTIME VTK OUTPUT
        OUTPUT_DATA_FORMAT                    binary
        INTERVAL_STEPS                        1
        EVERY_ITERATION                       no
        ----------------------------------------IO/RUNTIME VTK OUTPUT/STRUCTURE
        OUTPUT_STRUCTURE                      yes
        DISPLACEMENT                          yes
        --------------------------------------------IO/RUNTIME VTK OUTPUT/BEAMS
        OUTPUT_BEAMS                          yes
        DISPLACEMENT                          yes
        USE_ABSOLUTE_POSITIONS                yes
        TRIAD_VISUALIZATIONPOINT              yes
        STRAINS_GAUSSPOINT                    yes
        ----------------------------------------------------------------BINNING STRATEGY
        BIN_SIZE_LOWER_BOUND                  3.0
        DOMAINBOUNDINGBOX                     -30 -30 -30 30 30 30
        ----------------------------------------------------------------BEAM INTERACTION
        REPARTITIONSTRATEGY                   Everydt
        SEARCH_STRATEGY                       bounding_volume_hierarchy
        """
    )

    return input_file


if __name__ == "__main__":
    """Execution part of script."""

    # Adapt this path to the directory you want to store the tutorial files in.
    output_directory = "/home_student/kayabek/sw/Results/arc_shape2/"
    input_file = create_beams_wrapped_around_cylinder(output_directory)
    input_file.write_input_file(os.path.join(output_directory, "simple_beam.dat"))

    simulation_dir = "/home_student/kayabek/sw/Results/arc_shape2/results"

    return_code = run_four_c(
        os.path.join(output_directory, "simple_beam.dat"),
        simulation_dir,
        output_name='xxx',
        n_proc=2 )