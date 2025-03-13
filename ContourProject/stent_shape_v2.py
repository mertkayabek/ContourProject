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
from meshpy import (
    mpy,	
    Mesh,
    MaterialReissner,
    Beam3rHerm2Line3,
    BoundaryCondition,
    Rotation,
    Function,
    GeometrySet,
    InputFile,
)
from meshpy.four_c import run_four_c
from meshpy.utility import get_single_node
from meshpy.utility import find_close_nodes
from meshpy.mesh_creation_functions import (
    create_beam_mesh_line,
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_curve,
)

from meshpy.four_c import run_four_c
def create_two_beams_wrapped_around_cylinder(base_dir, preview=False):

    # use find_close_nodes() instead of this func

    def find_intersections_and_apply_coupling(mesh, beams):
        """
        Find intersections between beams and apply coupling.
        """
        """
        node_sets = [beam["line"].get_all_nodes() for beam in beams]
        intersecting_nodes = set() # make this a geometry set

        # Find intersecting nodes
        for i in range(len(node_sets)):
            for j in range(i + 1, len(node_sets)):
                for node_i in node_sets[i]:
                    for node_j in node_sets[j]:
                        if np.allclose(node_i.coordinates, node_j.coordinates):
                            intersecting_nodes.add(node_i)
                            intersecting_nodes.add(node_j)

        # Apply coupling to intersecting nodes
        
        for node_i, node_j in intersecting_nodes:
            mesh.couple_nodes(
                nodes=[node_i, node_j],
                coupling_type=mpy.bc.point_coupling_penalty,
                coupling_dof_type="POSITIONAL_PENALTY_PARAMETER 200 ROTATIONAL_PENALTY_PARAMETER 0"
            )
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
            print("\nLevel1:")
            if len(node_group) > 1:  # Only process groups with multiple nodes
                print("\nLevel2:")
                # Add nodes to geometry set
                for node in node_group:
                    intersecting_nodes.add(node)
                
                # Apply coupling between nodes in this group
                mesh.couple_nodes(
                    nodes=[node_group(0), node_group(1)],
                    coupling_type=mpy.bc.point_coupling_penalty,
                    coupling_dof_type="POSITIONAL_PENALTY_PARAMETER 200 ROTATIONAL_PENALTY_PARAMETER 0"
                )
        
        return intersecting_nodes

    mesh = Mesh()

    mat = MaterialReissner(youngs_modulus=200, radius=0.02)
    beam_object = Beam3rHerm2Line3

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
    
    def create_multiple_beams_shifted_in_y(
        mesh,
        number_of_beams,
        cylinder_radius,
        interval,
        n_el,
        add_sets,
        material,
        beam_object
    ):
        """
        Creates multiple beams, each shifted in y by an equal amount of 2π/number_of_beams.
        """
        """
        curve_radius = 0.05
        def n_shape_yz(t):
            x = cylinder_radius
            # Calculate total length and adjust curve parameters
            total_length = interval[1] - interval[0]
            curve_start = interval[1] - curve_radius
            
            if t <= curve_start:
                # Straight part
                y = 4.0 * t
                z = 1.0 * t
            else:
                # Curved part
                straight_y = 4.0 * curve_start
                straight_z = 1.0 * curve_start
                
                # Normalize parameter for curve
                t_curve = (t - curve_start) / curve_radius
                angle = npAD.pi/2 * t_curve
                
                # Create quarter-circle curve
                y = straight_y - curve_radius * (1 - npAD.cos(angle))
                z = straight_z + curve_radius * npAD.sin(angle)
            
            return npAD.array([x, y, z])
        
        def n_shape_yz2(t):
            x = cylinder_radius
            # Use same parameters as first beam for symmetry
            total_length = interval[1] - interval[0]
            curve_start = interval[1] - curve_radius
            
            if t <= curve_start:
                # Straight part (mirror of first beam)
                y = -4.0 * t
                z = 1.0 * t
            else:
                # Curved part
                straight_y = -4.0 * curve_start
                straight_z = 1.0 * curve_start
                
                # Normalize parameter for curve
                t_curve = (t - curve_start) / curve_radius
                angle = npAD.pi/2 * t_curve
                
                # Create quarter-circle curve (mirror of first beam)
                y = straight_y + curve_radius * (1 - npAD.cos(angle))
                z = straight_z + curve_radius * npAD.sin(angle)
            
            return npAD.array([x, y, z])
        """
        def n_shape_yz(t):
            x = cylinder_radius
            y = 4.0 * t
            z = 1.0 * t
            return npAD.array([x, y, z])
        
        def n_shape_yz2(t):
            x = cylinder_radius
            y = - 4.0 * t
            z = 1.0 * t
            return npAD.array([x, y, z])

        beams = []
        beams_start = []
        for i in range(number_of_beams):
            shift_i = (2.0 * npAD.pi / number_of_beams) * i
            
            def shape_with_shift(t, shift=shift_i):
                base = n_shape_yz(t)
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

            def shape_with_shift2(t, shift=shift_i):
                base = n_shape_yz2(t)
                return npAD.array([base[0], base[1] + shift, base[2]])

            dir2 = create_beam_mesh_curve(
                mesh,
                beam_object,
                material,
                shape_with_shift2,
                interval=interval,
                n_el=n_el,
                add_sets=add_sets
            )
            beams.append(dir1)
            beams.append(dir2)
            beams_start.append(dir1["start"])
            beams_start.append(dir2["start"])
            # apply start BC to beams_start

        
        #mesh.display_pyvista()
        mesh.wrap_around_cylinder(radius=cylinder_radius)
        mesh.display_pyvista()
        find_intersections_and_apply_coupling(mesh, beams) # change this
        mpy.check_overlapping_elements = False

        print("\nBeam end point coordinates:")
        for i, beam in enumerate(beams):
            start_node = beam["start"].get_all_nodes()[0]  # Get the first node from the geometry set
            start_coords = start_node.coordinates
            print(f"Beam {i + 1}: ({start_coords[0]:.3f}, {start_coords[1]:.3f}, {start_coords[2]:.3f})")
            end_node = beam["end"].get_all_nodes()[0]  # Get the first node from the geometry set
            end_coords = end_node.coordinates
            print(f"Beam {i + 1}: ({end_coords[0]:.3f}, {end_coords[1]:.3f}, {end_coords[2]:.3f})")
        
        # Define time steps
        num_steps = 10
        time_steps = np.linspace(0, 1.0, num_steps)
        compression_factor = 0.4  # Maximum compression
        
        # Create time function
        time_function = Function(
            "SYMBOLIC_FUNCTION_OF_SPACE_TIME t"
        )
        mesh.add(time_function)

        # write a for loop for all nodes in mesh like below
        """
        for node in mesh.nodes:
            if not node.is_middle
        """
        for node in mesh.nodes:

            if not node.is_middle:

                if node in beams_start:
                    # add here the boundary condition with radial and axial=0 displacement


                elif node.coordinates[2]>0:
                    # add here the boundary condition with radial displacement according to coordinate

                else:
                    pass

        # add here the coupling conditions 

        for beam in beams:

            # Fix start node
            mesh.add(
                BoundaryCondition(
                    beam["start"],
                    (
                        "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 "
                        "VAL 0 0 0 0 0 0 0 0 0 "
                        "FUNCT 0 0 0 0 0 0 0 0 0"
                    ),
                    bc_type=mpy.bc.dirichlet,
                )
            )

            # Get all nodes and apply force to first half
            all_nodes = beam["line"].get_all_nodes()
            middle_index = len(all_nodes) // 2

            # Apply compressive force to end node
            """
            end_node = beam["end"].get_all_nodes()[0]
            x, y, z = end_node.coordinates
            r = np.sqrt(x**2 + y**2)
            force_x = x / r * load_val
            force_y = y / r * load_val
            """
            load_val = -0.4
            for node in all_nodes[1:middle_index]:
                #if node.is_middle():
                x, y, z = node.coordinates
                r_initial = np.sqrt(x**2 + y**2)
                
                # Create lists to store displacements for each time step
                displacements = []
                for t in time_steps:
                    # Calculate target radius for this time step
                    current_compression = compression_factor * t
                    new_radius = cylinder_radius * (1.0 - current_compression)
                    
                    # Calculate displacement for this time step
                    displacement = calculate_displacement_for_cylinder(
                        node.coordinates, 
                        new_radius
                    )
                    displacements.append(displacement)

                node_set = GeometrySet(node)

                # Create displacement functions with time interpolation
                displacement_x = Function(
                    "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                    "VARIABLE 0 NAME a TYPE linearinterpolation "
                    "NUMPOINTS 3 TIMES 0 1 1000 VALUES 0 1 1".format(
                        len(time_steps),
                        " ".join(map(str, time_steps)),
                        " ".join(map(str, [d[0] for d in displacements]))
                    )
                )
                
                displacement_y = Function(
                    "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                    "VARIABLE 0 NAME a TYPE linearinterpolation "
                    "NUMPOINTS {} TIMES {} VALUES {}".format(
                        len(time_steps),
                        " ".join(map(str, time_steps)),
                        " ".join(map(str, [d[1] for d in displacements]))
                    )
                )
                
                mesh.add(displacement_x)
                mesh.add(displacement_y)
                
                # Apply boundary condition with calculated displacements
                mesh.add(
                    BoundaryCondition(
                        node_set,
                        (
                            "NUMDOF 9 ONOFF 1 1 0 0 0 0 0 0 0 "
                            "VAL {} {} 0 0 0 0 0 0 0 "
                            "FUNCT {} {} 0 0 0 0 0 0 0"
                        ),
                        format_replacement=[load_val, load_val, generic_linear_inpterol, generic_linear_inpterol],
                        bc_type=mpy.bc.dirichlet,
                    )
                )
            

    create_multiple_beams_shifted_in_y(
        mesh,
        2,
        1.0,
        [1, 11],
        50,
        True,
        mat,
        beam_object
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
        """
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
        TIMESTEP                              0.1
        NUMSTEP                               10
        MAXTIME                               1.0
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
        DOMAINBOUNDINGBOX                     -5 -5 -5 5 5 5
        ----------------------------------------------------------------BEAM INTERACTION
        REPARTITIONSTRATEGY                   Everydt
        SEARCH_STRATEGY                       bounding_volume_hierarchy
        """
    )

    return input_file


if __name__ == "__main__":
    """Execution part of script."""

    # Adapt this path to the directory you want to store the tutorial files in.
    output_directory = "/home_student/kayabek/sw/meshpy/ContourProject/results_stentshape/"
    input_file = create_two_beams_wrapped_around_cylinder(output_directory)
    input_file.write_input_file(os.path.join(output_directory, "simple_beam.dat"))

    simulation_dir = "/home_student/kayabek/sw/meshpy/ContourProject/results_stentshape/results"

    return_code = run_four_c(
        os.path.join(output_directory, "simple_beam.dat"),
        simulation_dir,
        output_name='xxx',
        n_proc=2 )