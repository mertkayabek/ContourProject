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
"""This script createss a flow diverter using beams."""

# Import python modules.
import os

import numpy as np

import yaml
from cubitpy import CubitPy, cupy
from cubitpy.mesh_creation_functions import extrude_mesh_normal_to_surface


from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.core.mesh import Mesh

# User modules
from meshpy.cosserat_curve.cosserat_curve import CosseratCurve
from meshpy.cosserat_curve.warping_along_cosserat_curve import (
    create_transform_boundary_conditions,
    warp_mesh_along_curve,
)
from meshpy.four_c.boundary_condition import BoundaryCondition
from meshpy.four_c.element_beam import Beam3rHerm2Line3
from meshpy.four_c.function import Function
from meshpy.four_c.function_utility import create_linear_interpolation_function
from meshpy.four_c.header_functions import (
    set_header_static,
    set_runtime_output,
)
from meshpy.four_c.input_file import InputFile, InputSection
from meshpy.four_c.material import MaterialReissner, MaterialStVenantKirchhoff
from meshpy.four_c.run_four_c import run_four_c, clean_simulation_directory
from meshpy.mesh_creation_functions.beam_basic_geometry import create_beam_mesh_helix
from meshpy.utils.nodes import check_node_by_coordinate
from meshpy.four_c.function import Function
from meshpy.core.rotation import Rotation


import autograd.numpy as npAD
from meshpy.utils.nodes import get_single_node, find_close_nodes
from meshpy.mesh_creation_functions.beam_basic_geometry import (
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_line,
)
from meshpy.mesh_creation_functions.beam_curve import create_beam_mesh_curve


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

def create_beams_wrapped_around_cylinder(
    cubit,
    second_simulation,
    preview=False,
    config=None,  # Add config parameter
):
    # Read simulation parameters from config file
    contour_config = config["contour"]
    disp_config = config["displacement"]
    time_config = config["time"]
    base_dir = contour_config["output_directory"] if "output_directory" in contour_config else "/home_student/kayabek/sw/meshpy/ContourProject/example_toy/cubit_results"  # Default value if not provided
    interval_end = contour_config["interval_end"] if "interval_end" in contour_config else 6.0  # Default value if not provided
    cylinder_radius = contour_config["cylinder_radius"] if "cylinder_radius" in contour_config else 3.0  # Default value if not provided
    compressed_part = contour_config["compressed_part"] if "compressed_part" in contour_config else 0.0  # Default value if not provided
    beam_radius = contour_config["beam_radius"] if "beam_radius" in contour_config else 0.015  # Default value if not provided
    youngs_modulus = contour_config["youngs_modulus"] if "youngs_modulus" in contour_config else 83000  # Default value if not provided
    positional_coupling_penalty = contour_config["pos_penalty"] if "pos_penalty" in contour_config else 100
    rotational_coupling_penalty = contour_config["rot_penalty"] if "rot_penalty" in contour_config else 0.01
    z_scale_factor = contour_config["z_scale_factor"] if "z_scale_factor" in contour_config else 0.5
    number_of_beams = contour_config["number_of_beams"] if "number_of_beams" in contour_config else 12
    compression_factor = contour_config["compression_factor"] if "compression_factor" in contour_config else 0.95  # Default value if not provided
    z_intermediate = disp_config["z_intermediate"] if "z_intermediate" in disp_config else -0.5
    z_final = disp_config["z_final"] if "z_final" in disp_config else -2.5

    num_steps = time_config["steps1"] if "steps1" in time_config else 100  # Default value if not provided
    time_step = 1/num_steps

    interval = [0, interval_end]
    n_intersections = 2
    n_el = 8*(n_intersections-1)*number_of_beams
    # austenite phase value
    # radius of marker max 0.25 mm

    mesh = Mesh()
    mat = MaterialReissner(youngs_modulus=youngs_modulus, radius=beam_radius)
    beam_object = Beam3rHerm2Line3


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
        close_node_groups = find_close_nodes(mesh.nodes)#, tol=1e-1)
        #print("\nFound close node groups:")
        print("Number of close node groups:", len(close_node_groups))
        for group_idx, node_group in enumerate(close_node_groups):
            if len(node_group) > 1:  # Only print groups with multiple nodes
                #print(f"\nGroup {group_idx + 1}:")
                for node_idx, node in enumerate(node_group):
                    coords = node.coordinates
                    #print(f"Node {node_idx + 1}: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})")
                #print("-" * 50)
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
        compressed_part,
        #node_positions=None  # Add this parameter
    ):
        """
        Creates multiple beams, each shifted in y by an equal amount of 2π/number_of_beams.
        """
    
        def n_shape_yz_complete(t):
            x = cylinder_radius
            R = interval[1]/2  # Radius of the arc (half of total interval)
            arc_height = R  # Maximum height of the arc
            
            # Parameter from 0 to π (half circle)
            theta = (t/(interval[1])) * npAD.pi
            
            # Calculate coordinates
            z = z_scale_factor * R * npAD.sin(theta)  # z goes up and down
            #z = R * npAD.sin(theta)  # z goes up and down
            y = tan_yz * R * (1 - npAD.cos(theta))  # y increases throughout
            
            return npAD.array([x, y, z])
        
        def calculate_node_positions(num_positions, R, tan_yz):
            """
            Calculate normalized parameter positions that correspond to uniform y-coordinate spacing.
            
            Args:
                number_of_beams: Number of beams (will create number_of_beams+1 nodes)
                R: Radius of the arc (half of total interval)
                tan_yz: Tangent angle factor for the arc
                
            Returns:
                List of normalized positions in [0,1] for node placement
            """
            # Start with position 0
            positions = [0.0]  # First node is at the start point
            
            # Calculate y-coordinate increment
            y_increment = 2 * tan_yz * R / number_of_beams
            
            # For each position, calculate the corresponding parameter value
            for i in range(1, number_of_beams):
                y = i * y_increment
                #print("y found:", y)
                #print("z found:", 0.5 * np.sqrt(2*R*y - y**2))
                # Invert the y-coordinate function to find the parameter t
                # y = tan_yz * R * (1 - cos(θ)) and θ = (t/interval[1]) * π
                # Solving for t: t = interval[1] * arccos(1 - y/(tan_yz*R)) / π
                #z = R * npAD.sin(theta)  # z goes up and down
                #y = tan_yz * R * (1 - npAD.cos(theta))  # y increases throughout
                
                # Handle potential numerical issues
                arc_length = calculate_elliptical_arc_length(y, R, tan_yz, z_scale_factor)
                half_ellipse_length = calculate_elliptical_arc_length(2*R, R, tan_yz, z_scale_factor)
                normalized_t = arc_length / half_ellipse_length
                positions.append(normalized_t)

            # End with position 1
            positions.append(1.0)

            #Now add 4 extra nodes between start (0.0) and first calculated node
            step = (positions[1] - positions[0]) / 4
            current = positions[0] + step  # Start at 0.01
            while current < positions[1]:
                if current not in positions:
                    positions.append(current)
                    positions.append(1 - current)
                #positions.sort()
                current += step
            
            # Sort the positions to maintain proper order7
            positions.sort()
            #print("Sorted node positions:", positions)
            return positions
        
        def calculate_elliptical_arc_length(target_y, R, tan_yz, z_scale_factor, num_samples=1000000):
            """
            Calculate the arc length along the elliptical curve from (y=0,z=0) to a specified y-value.
            
            Args:
                target_y: The y-coordinate to calculate arc length to
                R: Radius parameter (interval[1]/2)
                tan_yz: Tangent factor for y-coordinate
                z_scale_factor: Scaling factor for z-coordinate
                num_samples: Number of samples for numerical integration
            
            Returns:
                Arc length from origin to the point with the specified y-coordinate
            """  
            # Find theta corresponding to the target y-value
            # From: y = tan_yz * R * (1 - cos(theta))
            arg = 1 - target_y / (tan_yz * R)
            
            # Handle potential numerical issues
            if arg < -1:
                arg = -1
            elif arg > 1:
                arg = 1
            
            target_theta = np.arccos(arg)

            # Create sample points for numerical integration
            theta_values = np.linspace(0, target_theta, num_samples)
            
            # Calculate arc length using numerical integration
            arc_length = 0
            
            for i in range(1, len(theta_values)):
                theta1 = theta_values[i-1]
                theta2 = theta_values[i]
                
                # Midpoint approximation
                theta_mid = (theta1 + theta2) / 2
                
                # Calculate the integrand at the midpoint
                # √[(dy/dθ)² + (dz/dθ)²] 
                dy_dtheta = tan_yz * R * np.sin(theta_mid)
                dz_dtheta = z_scale_factor * R * np.cos(theta_mid)
                
                integrand = np.sqrt(dy_dtheta**2 + dz_dtheta**2)
                
                # Add segment length
                arc_length += integrand * (theta2 - theta1)

            #print("Target theta (degrees):", np.degrees(target_theta))
            #print("Arc length:", arc_length)
            return arc_length

        # Calculate node positions - number_of_beams/2 + 2 positions (including start and end)
        num_positions = number_of_beams // 2
        # Calculate node positions - number_of_beams+1 positions (including start and end)
        node_positions = calculate_node_positions(number_of_beams, interval[1]/2, tan_yz)
        
        #print("Node positions:", node_positions)

        beams = []
        beams_start = []
        beams_end = []
        
        for i in range(number_of_beams):
            #shift_i = (2.0 * npAD.pi * cylinder_radius / number_of_beams) * i
            shift_i = (2*(interval[1] - interval[0])/number_of_beams) * i
            #shift_i = (2*(interval[1] - interval[0])/ npAD.pi) * i
            def shape_with_shift(t, shift=shift_i):
                base = n_shape_yz_complete(t)
                return npAD.array([base[0], base[1] + shift, base[2]])

            dir1 = create_beam_mesh_curve(
                mesh,
                beam_object,
                material,
                shape_with_shift,
                interval=interval,
                #n_el=n_el,
                node_positions_of_elements=node_positions,  # Use calculated positions instead of n_el
                add_sets=add_sets
            )

            beams.append(dir1)
            beams_start.append(dir1["start"])
            beams_end.append(dir1["end"])
            # Add this: Assign beam number to all nodes in this beam
            beam_nodes = dir1["line"].get_all_nodes()
            """
            for node in beam_nodes:
                node.beam_number = i  # Store the beam index as an attribute
                print(f"Beam {i}: x = {node.coordinates[0]:.3f}, y = {node.coordinates[1]:.3f}, z = {node.coordinates[2]:.3f}")
            """            
    
            if i == 0:
                #print("\nY-coordinates of nodes on first arc:")
                beam_nodes = dir1["line"].get_all_nodes()
                beam_nodes.sort(key=lambda node: node.coordinates[1])
                # for j, node in enumerate(beam_nodes):
                # print(f"Node {j}: y = {node.coordinates[1]:.6f}")

        mesh.wrap_around_cylinder(radius=cylinder_radius)
        find_intersections_and_apply_coupling(mesh, beams, positional_coupling_penalty, rotational_coupling_penalty) # change this
        mpy.check_overlapping_elements = False     

        new_radius = cylinder_radius * (1.0 - compression_factor)

        # write a for loop for all nodes in mesh like below
        max_z = max(node.coordinates[2] for node in mesh.nodes if not node.is_middle_node)
        threshold = max_z * compressed_part
        num_nodes = len(mesh.nodes)
        
        #beam_nodes_for_contact = GeometrySet()
        #beam_nodes_for_contact = None
        beam_lines = GeometrySet(beams[0]["line"])
        for i in range(1, len(beams)):
            beam_lines.add(beams[i]["line"])

        for node in mesh.nodes:
            #print(f"Beam {beam_num}: x = {node.coordinates[0]:.3f}, y = {node.coordinates[1]:.3f}, z = {node.coordinates[2]:.3f}")
            #print(f"Beam: x = {node.coordinates[0]:.3f}, y = {node.coordinates[1]:.3f}, z = {node.coordinates[2]:.3f}")
            if not node.is_middle_node:

                if  np.linalg.norm(node.coordinates[2] - interval[0]) < 1e-9:                #node in beams_start:
                    #print(f"Start node coordinates: {node.coordinates}")
                    # boundary condition with radial and axial=0 displacement
                    node_set = GeometrySet(node)
                    #print(f"Start Node coordinates: {node.coordinates}")
                    
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
                    displacement_z = Function(
                        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                        "VARIABLE 0 NAME a TYPE linearinterpolation "
                        "NUMPOINTS 4 TIMES 0.0 1.0 2.0 1000.0 VALUES 0.0 {} {} {}".format(
                            z_intermediate,
                            z_final,
                            z_final
                        )
                    )

                    mesh.add(displacement_x)
                    mesh.add(displacement_y)
                    mesh.add(displacement_z)

                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 "  # Fix z also only x and y translations
                                "VAL 1 1 1 0 0 0 0 0 0 "
                                "FUNCT {} {} {} 0 0 0 0 0 0"  # Use displacement functions for x,y
                            ),
                            format_replacement=[displacement_x, displacement_y, displacement_z],  # Use the displacement functions
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                    
                
                elif node.coordinates[2] <= threshold:
                #node.coordinates[2] < ((interval[1] + interval[0])/2)*compressed_part:  # z < 5 for interval [0, 10]
                    # add here the boundary condition with radial displacement according to coordinate
                    node_set = GeometrySet(node)

                    displacement = calculate_displacement_for_cylinder(
                            node.coordinates, 
                            new_radius
                        )
                    #print(f"Compressed Node coordinates: {node.coordinates[2]}")
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
                                # no need for fixing rotation check that
                                "NUMDOF 9 ONOFF 1 1 0 0 0 0 0 0 0 "  # Fix only x and y translations
                                "VAL 1 1 0 0 0 0 0 0 0 "
                                "FUNCT {} {} 0 0 0 0 0 0 0"  # Use displacement functions for x,y
                            ),
                            format_replacement=[displacement_x, displacement_y],  # Use the displacement functions
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                    
                    
        if second_simulation:
            mesh.add(
                BoundaryCondition(
                    beam_lines,
                    "COUPLING_ID 2",
                    bc_type=mpy.bc.beam_to_solid_surface_contact,
                )
            )  
    tan_yz, degrees = calculate_angle_for_intersections(n_intersections, interval, cylinder_radius)
    
    #print(f"degrees: {degrees}")
    interval[1] = interval[1] * tan_yz
    tan_yz = 1

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
    input_file = InputFile(cubit=cubit)

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
        OUTPUT_BIN                            yes
        STRUCT_DISP                           yes
        FILESTEPS                             1
        VERBOSITY                             Standard
        STRUCT_STRAIN                         yes
        STRUCT_STRESS                         yes
        -----------------------------------------------------STRUCTURAL DYNAMIC
        LINEAR_SOLVER                         1
        INT_STRATEGY                          Standard
        DYNAMICTYPE                           Statics
        RESULTSEVERY                          1
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
        ELEMENT_GID                           yes
        ----------------------------------------------------------------BINNING STRATEGY
        BIN_SIZE_LOWER_BOUND                  10.0
        DOMAINBOUNDINGBOX                     -30 -30 -30 30 30 30
        ----------------------------------------------------------------BEAM INTERACTION
        REPARTITIONSTRATEGY                   Everydt
        SEARCH_STRATEGY                       bounding_volume_hierarchy
        """
    )

    return input_file


# global settings
preview = False



def setup_pre_deformed_flow_diverter(cubit, config, second_simulation):
    """Creates a flow diverter model which is compressed into a tube with
    different diameter.
    replace me with contour device!

    Args
    """

    # set up a mesh
    mesh_flow_diverter = Mesh()

    # material
    mat = MaterialReissner(youngs_modulus=116000, radius=config["fd"]["r_wire"])

    # beam
    beam_object = Beam3rHerm2Line3

    # load parameters from config file
    L = config["fd"]["L"]
    r_fd = config["fd"]["r_fd"]
    n_turns = config["fd"]["n_turns"]
    n_wire = config["fd"]["n_wire"]

    # set the pre compressed diameter of
    fd_pre_r = config["catheter"]["fd_pre_r"] - 0.01

    # coupling_nodes=[]
    # create a mesh consisting of multiple helix for a fd
    for clockwise in [-1.0, 1]:
        for i in range(1, n_wire + 1):
            # create beams according the flow diverter function
            beam_set = create_beam_mesh_helix(
                mesh_flow_diverter,
                beam_object,
                mat,
                axis_vector=[0.0, 0.0, clockwise],
                axis_point=[0.0, 0.0, 0.0],
                start_point=[
                    r_fd * np.cos(i * 2 * np.pi / n_wire),
                    r_fd * np.sin(i * 2 * np.pi / n_wire),
                    0.0,
                ],
                helix_angle=-clockwise * np.arctan(L / (2 * r_fd * np.pi * n_turns)),
                height_helix=L,
                n_el=config["fd"]["n_ele_beam"],
            )
            if "coupling_nodes" in locals():
                coupling_nodes.add(beam_set["line"])
            else:
                coupling_nodes = beam_set["line"]

            # apply boundary conditions directly on nodes
            for i, node in enumerate(beam_set["line"].get_all_nodes()):
                # do not constraint middle nodes
                if not node.is_middle_node:
                    # get pre displacement

                    displacement_zylinder = calculate_displacement_for_cylinder(
                        node.coordinates, fd_pre_r
                    )

                    # calculate the displacement
                    # this calculation for x and y needs to be replaced by local sys condition
                    # since for complex geometries this is not a circle any more!

                    # Set up Condition for all nodes which ly on the z-plane of the origin
                    # (one side of the cylinder)
                    if check_node_by_coordinate(node, 2, 0):
                        mesh_flow_diverter.add(
                            BoundaryCondition(
                                GeometrySet(node),
                                "NUMDOF 9 ONOFF 0 0 1 1 1 1 0 0 0 VAL {} {} 0 0 0 0 0 0 0 FUNCT 1 1 0 0 0 0 0 0 0 TAG monitor_reaction ".format(
                                    displacement_zylinder[0],
                                    displacement_zylinder[1],
                                ),
                                bc_type=mpy.bc.dirichlet,
                            )
                        )
                    # for restarted simulation only constraint axial direction
                    # if check_node_by_coordinate(node,2,0) and second_simulation:
                    #    mesh_flow_diverter.add(
                    #        BoundaryCondition(
                    #        GeometrySet(node),
                    #        "NUMDOF 9 ONOFF 0 1 1 0 0 0 0 0 0 VAL 0 {} 0 0 0 0 0 0 0 FUNCT 0 1 0 0 0 0 0 0 0 TAG monitor_reaction".format(),
                    #        bc_type=mpy.bc.dirichlet,
                    #       )
                    #   )

    mesh_flow_diverter.add(
        BoundaryCondition(
            coupling_nodes,
            "COUPLING_ID 1",
            bc_type=mpy.bc.beam_to_solid_surface_contact,
        )
    )
    #TODO: Add me to the simulation
    if second_simulation:
        mesh_flow_diverter.add(
            BoundaryCondition(
                coupling_nodes,
                "COUPLING_ID 2",
                bc_type=mpy.bc.beam_to_solid_surface_contact,
            )
        )

    # translate the mesh to the middle
    mesh_flow_diverter.translate([0, 0, L / 2])

    mesh_flow_diverter.couple_nodes(
        reuse_matching_nodes=True,
        coupling_type=mpy.bc.point_coupling_penalty,
        coupling_dof_type="POSITIONAL_PENALTY_PARAMETER 10000 ROTATIONAL_PENALTY_PARAMETER 0",  # noqa
    )

    #if preview:
        #mesh_flow_diverter.display_pyvista()

    # TODO: MERT
    # create Input File form cubit
    input_file = InputFile(cubit=cubit)

    # Add monitoring
    input_file.add(
        """
        --IO/MONITOR STRUCTURE DBC
        PRECISION_FILE         10
        PRECISION_SCREEN       5
        FILE_TYPE              csv
        WRITE_HEADER           yes
        INTERVAL_STEPS         1
        """
    )
    input_file.add("ELEMENT_MAT_ID                  yes")

    # Add the beam geometry to the input file.
    input_file.add(mesh_flow_diverter)

    # Define the 4c simulation header
    set_header_static(
        input_file,
        time_step=config["time"]["dt_1"] / config["time"]["steps1"],
        n_steps=config["time"]["steps1"],
        write_stress="Yes",
        max_iter=30,
        tol_residuum=1e-6,
        tol_increment=1e-6,
    )

    # Add additional function to apply displacements
    # TODO replace this
    # this is a factor such that the artery is not penatrated by the wall.
    # so instead of applying this dirichlet condition until the initial state we implicitly
    # calculate the end point with this factor
    # however this should be determined by the contact simulation rather by a pre determined calculation.
    r_factor_final = (config["fd"]["r_fd"] - config["catheter"]["fd_pre_r"]) - (
        config["fd"]["r_fd"] - config["aneurysm"]["r_art_i"]
    )
    fun = Function(
        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a \nVARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 4 TIMES 0 1 2 9999999999.0 VALUES 0.0 1.0 {} 0.0".format(
            r_factor_final
        )
    )
    input_file.add(fun)

    # Add Runtime options
    set_runtime_output(input_file, output_stress_strain=True)
    input_file.add("--IO\nOUTPUT_BIN yes\nSTRUCT_DISP yes", option_overwrite=True)

    # TODO: MERT add this
    input_file.add(
        """----------------------------------BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT
                    CONSTRAINT_STRATEGY                      penalty
                    CONTACT_DISCRETIZATION                   mortar
                    CONTACT_TYPE                             gap_variation
                    GEOMETRY_PAIR_SEGMENTATION_SEARCH_POINTS 2
                    GAUSS_POINTS                             6
                    GEOMETRY_PAIR_STRATEGY                   segmentation
                    PENALTY_LAW                              linear_quadratic
                    PENALTY_PARAMETER                        400.0
                    PENALTY_PARAMETER_G0                     0.0001
                    MORTAR_SHAPE_FUNCTION                    line2
                    MORTAR_CONTACT_DEFINED_IN                reference_configuration
                """
    )
    input_file.add(
        """----------------------------------------------------------------BEAM INTERACTION
        REPARTITIONSTRATEGY                   Everydt
        --------------------------------------------------------------------BEAM CONTACT
        MODELEVALUATOR                        Standard
        ----------------------------------------------------------------BINNING STRATEGY
        BIN_SIZE_LOWER_BOUND                  5.0
        DOMAINBOUNDINGBOX -10 -10 -20 10 10 20
        """
    )

    return input_file


def create_straight_toy_aneurysm(cubit, config, restart):
    """creates the straight toy aneurysm with a hex8 mesh"""
    
    L = config["L"]
    r_art_i = config["r_art_i"]
    t = config["t"]
    r_ca_i = config["r_ca_i"]
    h_y_ca = config["h_y_ca"]
    n_ref_surf = config["n_ref_surf"]
    n_ref_neck = config["n_ref_neck"]

    if r_art_i + 2 * t + r_ca_i < h_y_ca:
        print("The specified height h_y_ca is to big. Can not create the geometry.")

    # Create inner and outer cylinder.
    cylinder_i = cubit.cylinder(L, r_art_i, r_art_i, r_art_i)

    sphere_i = cubit.sphere(r_ca_i)
    lumen = cubit.get_last_id("volume")

    group_id = cubit.create_new_group()
    cubit.add_entity_to_group(group_id, lumen, "volume")
    cubit.cmd(f"volume {sphere_i.id()} move 0 {h_y_ca} 0 ")
    cubit.cmd(f"unite {sphere_i.id()} {cylinder_i.id()}")

    united_volumes = cubit.get_last_id("volume")
    cubit.cmd(f"move volume {united_volumes} Y -1.5 Z 0")
    
    cubit.cmd(f"Rotate Volume {united_volumes} about X Angle 90")
    

    # TODO for Martin: replace numbers with algs to find curves and surfaces
    curvestotweak = [3]
    for curve_id in curvestotweak:
        cubit.cmd(f"Tweak Curve {curve_id} Fillet Radius {2*t}")
    
    print("surf_ids:", cubit.get_group_surfaces(group_id))
    cubit.cmd(f"surface {5} {6} {7} Scheme Auto")
    cubit.cmd("surface {5} {6} {7} size auto factor 4")
    cubit.cmd(f"mesh surface {5} {6} {7}")

    for n in range(int(n_ref_surf)):
        cubit.cmd(f"refine surface {5} {6} {7}")

    outer_surfaces = []
    cubit.cmd(f"merge surface {5} {6} {7}")

    for surf in cubit.volume(cubit.get_last_id(cupy.geometry.volume)).surfaces():
        if surf.id() in [5, 6, 7]:
            outer_surfaces.append(surf)
    extrude_mesh_normal_to_surface(
        cubit, outer_surfaces, t, n_layer=1, average_normals=1
    )

    wall_id = 1

    # Add block block
    cubit.add_element_type(
        cubit.group(add_value=f"add volume {wall_id}"),
        cupy.element_type.hex8,
        name="arterial wall",
        material="MAT 3",
        bc_description="KINEM nonlinear",
    )

    # Apply Boundary conditions on
    for surf in cubit.volume(wall_id).surfaces():
        # constraint for inner cylinder
        # identify surface over area this is a rough estimate and may change
        # todo find better way to identify some points(second biggest surface?)
        # TODO for Martin
        inner_surf=4
        if surf.id()==inner_surf and restart:
            cubit.add_node_set(
                cubit.group(add_value=f"add surface {surf.id()}"),
                name="CONTACT SURFACE 2 - ART ",
                bc_section="BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE",
                bc_description="COUPLING_ID 2",
            )

    # constrain only the artery outside
    outer_ca_surf=9

    cubit.add_node_set(
        cubit.group(add_value=f"add surface {outer_ca_surf}"),
        name="fix axial direction artery {}".format(outer_ca_surf),
        bc_section="DESIGN SURF DIRICH CONDITIONS",
        bc_description="NUMDOF 6 ONOFF 1 1 1 0 0 0  VAL 0 0 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0 0 0",
    )
    #cubit.display_in_cubit()

def setup_simulation(config, second_simulation):
    """
    returns an input file based yml configuration and first or second simulation.
    """

    # Initialize cubit
    cubit = CubitPy()

    # create a artery mesh (zylinder)
    create_straight_toy_aneurysm(cubit, config["aneurysm"], second_simulation)

    # display output
    if preview:
        cubit.display_in_cubit()

    # create first simulation
    # TODO: MERT create function which creates the contour device with boundary conditions
    input_file = create_beams_wrapped_around_cylinder(
        cubit,
        second_simulation,
        config=config,  # Add config parameter
    )

    # add missing material for artery
    mat_mc = MaterialStVenantKirchhoff(youngs_modulus=116000.0, nu=0.3, density=4506000)
    input_file.add(mat_mc)
    mat_artery = MaterialStVenantKirchhoff(
        youngs_modulus=11600.00, nu=0.3, density=4506000
    )
    input_file.add(mat_artery)

    #input_file.display_pyvista()

    return input_file


if __name__ == "__main__":
    """Execution part of script."""

    # read config file
    pwd = os.path.join(os.getcwd())
    with open(os.path.join(pwd, "/home_student/kayabek/sw/meshpy/ContourProject/example_toy/config.yml"), "r") as file:
        config = yaml.safe_load(file)

    # Adapt this path to the directory you want to store the simulation files
    simulation_dir_1 = "/home_student/kayabek/sw/meshpy/ContourProject/example_toy/cubit_results/simulation_step_1"
    os.makedirs(simulation_dir_1, exist_ok=True)
    
    inputfile_1 = setup_simulation(config, False)

    # ensure clean simulation directory
    clean_simulation_directory(simulation_dir_1)

    # write simulation file
    fd_placement_dat = os.path.join(simulation_dir_1, "fd-vmc-art.dat")
    inputfile_1.write_input_file(fd_placement_dat)
    run_four_c( fd_placement_dat, simulation_dir_1)

    simulation_dir_2 = "/home_student/kayabek/sw/meshpy/ContourProject/example_toy/cubit_results/simulation_step_2"
    os.makedirs(simulation_dir_2, exist_ok=True)

    # ensure clean simulation directory
    clean_simulation_directory(simulation_dir_2)

    fd_placement_dat2 = os.path.join(simulation_dir_2, "fd-vmc-art.dat")

    inputfile_1 = setup_simulation(config, True)
    
    inputfile_1.add(
        InputSection(
            "STRUCTURAL DYNAMIC",
            f"""
        LINEAR_SOLVER     1
        NUMSTEP           {config['time']['steps1']+config['time']['steps2']}
        MAXTIME           {config['time']['dt_1']+config['time']['dt_2']}
        TIMESTEP          {config['time']['dt_2']/config['time']['steps2']}
        """,
            option_overwrite=True,
        )
    )

    inputfile_1.add(
        """----------------------------------BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT
                    CONSTRAINT_STRATEGY                      penalty
                    CONTACT_DISCRETIZATION                   mortar
                    CONTACT_TYPE                             gap_variation
                    GEOMETRY_PAIR_SEGMENTATION_SEARCH_POINTS 4
                    GAUSS_POINTS                             4
                    GEOMETRY_PAIR_STRATEGY                   segmentation
                    PENALTY_LAW                              linear_quadratic
                    PENALTY_PARAMETER                        100.0
                    PENALTY_PARAMETER_G0                     0.0001
                    MORTAR_SHAPE_FUNCTION                    line2
                    MORTAR_CONTACT_DEFINED_IN                reference_configuration
                """
    )

    inputfile_1.write_input_file(fd_placement_dat2)
    run_four_c(
        fd_placement_dat2,
        simulation_dir_2,
        restart_step=(config["time"]["steps1"]),
        restart_from=os.path.join(os.path.relpath(simulation_dir_1, simulation_dir_2),"xxx"),
    )

    # subprocess.run(["/imcs/public/compsim/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/bin/paraview", "--state=paraview_state.pvsm"])