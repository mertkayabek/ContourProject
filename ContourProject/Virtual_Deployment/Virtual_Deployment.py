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
Medical Contour Device Simulation with Aneurysm Interaction

This script creates a comprehensive simulation of a medical contour device
with parametric aneurysm geometry, including:
1. Contour device beam structure generation
2. Parametric aneurysm surface creation  
3. Beam-to-beam interaction simulation
4. Beam-to-surface contact modeling

The contour device uses multiple helical beams arranged in a specific pattern
to provide medical treatment coverage for aneurysms.

The simulation is performed in two steps:
1. Device undeployed shape simulation
2. Device-aneurysm contact simulation

Author: Mert Kayabek
"""

# Standard library imports
import os
import numpy as np
import autograd.numpy as npAD
import yaml

# Cubit interface imports
from cubitpy import CubitPy, cupy
from cubitpy.mesh_creation_functions import extrude_mesh_normal_to_surface

# MeshPy core modules
from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.core.mesh import Mesh

# MeshPy Cosserat curve modules
from meshpy.cosserat_curve.cosserat_curve import CosseratCurve
from meshpy.cosserat_curve.warping_along_cosserat_curve import (
    create_transform_boundary_conditions,
    warp_mesh_along_curve,
)

# MeshPy 4C interface modules
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

# MeshPy mesh creation modules
from meshpy.mesh_creation_functions.beam_basic_geometry import (
    create_beam_mesh_helix,
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_line,
)
from meshpy.mesh_creation_functions.beam_curve import create_beam_mesh_curve

# MeshPy utility modules
from meshpy.utils.nodes import check_node_by_coordinate, get_single_node, find_close_nodes
from meshpy.core.rotation import Rotation


def calculate_displacement_for_cylinder(coordinates, new_radius):
    """
    Calculate radial displacement for cylindrical compression transformation.
    
    This function computes the displacement required to compress a point
    from its current radial position to a new radial position on a cylinder.
    
    Args:
        coordinates (array): [x, y, z] coordinates of the point
        new_radius (float): Target radius for compression
        
    Returns:
        array: [dx, dy, dz] displacement vector (dz = 0)
    """
    x = coordinates[0]
    y = coordinates[1]
    
    # Calculate current radial distance from z-axis
    original_radius = np.sqrt(x**2 + y**2)
    
    # Avoid division by zero for points on the axis
    if original_radius < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    
    # Calculate scaling factor for radial compression
    scaling_factor = new_radius / original_radius
    
    # Apply radial scaling to x and y coordinates
    new_x = x * scaling_factor
    new_y = y * scaling_factor
    
    # Return displacement (no z-displacement for cylindrical compression)
    return np.array([new_x - x, new_y - y, 0.0])


def create_beams_wrapped_around_cylinder(cubit, second_simulation, preview=False, config=None):
    """
    Create a contour device beam structure wrapped around a cylindrical surface.
    
    This function generates multiple helical beams that form a medical contour device.
    The beams are arranged to intersect and provide specific coverage patterns
    for medical treatment applications.
    
    Args:
        cubit: CubitPy interface object for mesh generation
        second_simulation (bool): Whether this is the second simulation step
        preview (bool): Whether to show preview visualization
        config (dict): Configuration dictionary with simulation parameters
        
    Returns:
        InputFile: Complete 4C input file for contour device simulation
    """
    # Extract configuration parameters with defaults
    contour_config = config["contour"]
    disp_config = config["displacement"]
    time_config = config["time"]
    
    # Output and geometry parameters
    default_output_dir = os.path.join(os.path.dirname(__file__), "Simulation")
    base_dir = contour_config.get("output_directory", default_output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    interval_end = contour_config.get("interval_end", 6.0)
    cylinder_radius = contour_config.get("cylinder_radius", 3.0)
    compressed_part = contour_config.get("compressed_part", 0.0)
    
    # Material properties
    beam_radius = contour_config.get("beam_radius", 0.015)
    youngs_modulus = contour_config.get("youngs_modulus", 83000)
    
    # Coupling parameters for beam interactions
    positional_coupling_penalty = contour_config.get("pos_penalty", 100)
    rotational_coupling_penalty = contour_config.get("rot_penalty", 0.01)
    
    # Geometric scaling factors
    z_scale_factor = contour_config.get("z_scale_factor", 0.5)
    number_of_wires = contour_config.get("number_of_wires", 12)
    compression_factor = contour_config.get("compression_factor", 0.95)

    z_intermediate = disp_config.get("z_intermediate", -1.0)
    z_final = disp_config.get("z_final", -2.5)
    
    # Time stepping parameters
    num_steps = time_config.get("steps1", 100)
    time_step = 1 / num_steps
    
    # Beam generation parameters
    interval = [0, interval_end]
    n_intersections = 2  # Number of beam intersections
    n_el = 8 * (n_intersections - 1) * number_of_wires  # Total number of elements
    
    # Initialize mesh and material
    mesh = Mesh()
    mat = MaterialReissner(youngs_modulus=youngs_modulus, radius=beam_radius)
    beam_object = Beam3rHerm2Line3

    def find_intersections_and_apply_coupling(mesh, beams, positional_coupling_penalty, rotational_coupling_penalty):
        """
        Find intersections between beams and apply penalty coupling.
        
        This function identifies nodes that are geometrically close between
        different beams and applies penalty-based coupling to simulate
        beam-to-beam interactions in the contour device structure.
        
        Args:
            mesh: Mesh object containing all beams
            beams: List of beam dictionaries
            positional_coupling_penalty: Penalty parameter for positional coupling
            rotational_coupling_penalty: Penalty parameter for rotational coupling
            
        Returns:
            GeometrySet: Set of intersecting nodes for visualization/analysis
        """
        # Find groups of nodes that are geometrically close
        close_node_groups = find_close_nodes(mesh.nodes)
        
        print(f"Found {len(close_node_groups)} close node groups for beam coupling")
        
        # Initialize geometry set with first intersecting group
        first_group = next((group for group in close_node_groups if len(group) > 1), None)
        if not first_group:
            print("Warning: No intersecting nodes found for beam coupling")
            return None
            
        intersecting_nodes = GeometrySet(first_group[0])
        
        # Apply penalty coupling for each group of close nodes
        coupling_count = 0
        for node_group in close_node_groups:
            if len(node_group) > 1:  # Only process groups with multiple nodes
                # Add all nodes to the geometry set for visualization
                for node in node_group:
                    intersecting_nodes.add(node)
                
                # Apply pairwise penalty coupling between first two nodes
                mesh.couple_nodes(
                    nodes=[node_group[0], node_group[1]],
                    coupling_type=mpy.bc.point_coupling_penalty,
                    coupling_dof_type=f"POSITIONAL_PENALTY_PARAMETER {positional_coupling_penalty} "
                                    f"ROTATIONAL_PENALTY_PARAMETER {rotational_coupling_penalty}"
                )
                coupling_count += 1
        
        print(f"Applied penalty coupling to {coupling_count} node groups")
        return intersecting_nodes

    def calculate_angle_for_intersections(n_intersections, interval, radius):
        """
        Calculate the helical angle required for desired number of beam intersections.
        
        This function determines the pitch angle needed for helical beams to
        intersect the specified number of times over the given interval length.
        
        Args:
            n_intersections (int): Desired number of intersections
            interval (list): [start, end] length of the beam in z axis
            radius (float): Cylinder radius
            
        Returns:
            tuple: (yz_ratio, alpha_degrees) - tangent and angle in degrees
        """
        length = interval[1] - interval[0]
        
        # Calculate required tangent for helical pitch
        # Intersections occur when beams complete (n_intersections-1) * π radians
        yz_ratio = ((n_intersections - 1) * np.pi * radius) / length
        alpha_degrees = np.degrees(np.arctan(yz_ratio))
        
        return yz_ratio, alpha_degrees

    def create_multiple_beams_shifted_in_y(mesh, number_of_wires, cylinder_radius, interval, n_el, add_sets, material, beam_object, compression_factor, compressed_part):
        """
        Create multiple helical beams with equal angular spacing around cylinder.
        
        This function generates the complete contour device structure by creating
        multiple helical beams, each rotated by 2π/number_of_wires around the
        cylinder axis. Each beam follows a parametric arc shape.
        
        Args:
            mesh: Mesh object to add beams to
            number_of_wires (int): Number of helical beams to create
            cylinder_radius (float): Radius of the cylindrical surface
            interval (list): [start, end] parametric interval
            n_el (int): Number of elements per wire
            add_sets (bool): Whether to add geometry sets
            material: Material object for beams
            beam_object: Beam element type
            compression_factor (float): Factor for beam compression
            compressed_part (float): Portion of wire that is compressed
            
        Returns:
            tuple: (beams, beams_start) - lists of beam objects and start nodes
        """
        
        def n_shape_yz_complete(t):
            """
            Define the parametric shape function for helical beam geometry.
            
            This creates a helical arc that starts at cylinder radius and
            follows a sinusoidal path in both y and z directions.
            
            Args:
                t (float): Parameter from 0 to interval[1]
                
            Returns:
                array: [x, y, z] coordinates of the beam curve
            """
            x = cylinder_radius  # Constant radial distance from z-axis
            R = interval[1] / 2  # Arc radius (half of total interval)
            
            # Parametric angle from 0 to π (half circle in yz-plane)
            theta = (t / interval[1]) * npAD.pi
            
            # Z-coordinate: creates vertical oscillation (scaled by z_scale_factor)
            z = z_scale_factor * R * npAD.sin(theta)
            
            # Y-coordinate: creates forward progression with helical component
            y = R * (1 - npAD.cos(theta))
            
            return npAD.array([x, y, z])
        
        def calculate_node_positions(num_positions, R):
            """
            Calculate optimal node positions for uniform y-coordinate spacing.
            
            This function determines parameter values that result in evenly
            spaced nodes along the y-direction, accounting for the nonlinear
            parametric curve shape.
            
            Args:
                num_positions (int): Number of positions to calculate
                R (float): Arc radius parameter                
            Returns:
                list: Normalized parameter positions [0,1] for node placement
            """
            positions = [0.0]  # Start at parameter t=0
            
            # Calculate uniform y-coordinate increments
            y_increment = 2 * R / number_of_wires
            
            # For each intermediate position, solve for the parameter value
            for i in range(1, number_of_wires):
                y_target = i * y_increment
                
                # Calculate arc length to this y-position (numerical integration)
                arc_length = calculate_elliptical_arc_length(y_target, R, z_scale_factor)
                half_ellipse_length = calculate_elliptical_arc_length(2 * R, R, z_scale_factor)
                
                # Normalize to [0,1] parameter range
                normalized_t = arc_length / half_ellipse_length
                positions.append(normalized_t)

            positions.append(1.0)  # End at parameter t=1
            
            # Add extra refinement nodes near the bottom of wires for better discretization
            step = (positions[1] - positions[0]) / 4
            current = positions[0] + step
            while current < positions[1]:
                if current not in positions:
                    positions.append(current)
                    positions.append(1 - current)  # Add symmetric position
                current += step
            
            positions.sort()
            return positions
        
        def calculate_elliptical_arc_length(target_y, R, z_scale_factor, num_samples=1000000):
            """
            Calculate arc length along the elliptical curve using numerical integration.
            
            This function computes the arc length from the curve origin to a point
            with the specified y-coordinate. Used for uniform node spacing along
            the curved beam geometry.
            
            Args:
                target_y (float): Target y-coordinate
                R (float): Arc radius parameter
                z_scale_factor (float): Z-direction scaling factor
                num_samples (int): Number of integration samples
                
            Returns:
                float: Arc length to the target y-coordinate
            """
            # Find theta corresponding to target y-coordinate
            arg = 1 - target_y / R
            
            # Handle numerical edge cases
            if arg < -1:
                arg = -1
            elif arg > 1:
                arg = 1
            
            target_theta = np.arccos(arg)
            
            # Numerical integration using trapezoidal rule
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
                dy_dtheta = R * np.sin(theta_mid)
                dz_dtheta = z_scale_factor * R * np.cos(theta_mid)
                
                integrand = np.sqrt(dy_dtheta**2 + dz_dtheta**2)
                
                # Add segment length
                arc_length += integrand * (theta2 - theta1)
            return arc_length

        # Calculate optimal node positions for beam discretization
        num_positions = number_of_wires // 2
        node_positions = calculate_node_positions(number_of_wires, interval[1] / 2)
        
        beams = []
        beams_start = []

        # Create multiple wires with angular spacing
        for i in range(number_of_wires):
            shift_i = (2*(interval[1] - interval[0])/number_of_wires) * i

            def shape_with_shift(t, shift=shift_i):
                base = n_shape_yz_complete(t)
                return npAD.array([base[0], base[1] + shift, base[2]])
            
            dir1 = create_beam_mesh_curve(
                mesh,
                beam_object,
                material,
                shape_with_shift,
                interval=interval,
                node_positions_of_elements=node_positions,  # Use calculated positions instead of n_el
                add_sets=add_sets
            )

            beams.append(dir1)
            beams_start.append(dir1["start"])

        mesh.wrap_around_cylinder(radius=cylinder_radius) # wrap mesh around cylinder
        find_intersections_and_apply_coupling(mesh, beams, positional_coupling_penalty, rotational_coupling_penalty)
        mpy.check_overlapping_elements = False 

        new_radius = cylinder_radius * (1.0 - compression_factor) # final radius after compression

        max_z = max(node.coordinates[2] for node in mesh.nodes if not node.is_middle_node)
        threshold = max_z * compressed_part # z coordinate threshold for compression

        beam_lines = GeometrySet(beams[0]["line"])
        for i in range(1, len(beams)):
            beam_lines.add(beams[i]["line"])

        for node in mesh.nodes:
            if not node.is_middle_node:
                
                # Nodes at the bottom
                if np.linalg.norm(node.coordinates[2] - interval[0]) < 1e-9:
                    node_set = GeometrySet(node)
                    
                    displacement = calculate_displacement_for_cylinder(
                            node.coordinates, 
                            new_radius
                        )
                    
                    # Compression is for x and y displacements
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

                    # The device is displaced in -z direction for deployment
                    displacement_z = Function(
                        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                        "VARIABLE 0 NAME a TYPE linearinterpolation "
                        "NUMPOINTS 4 TIMES 0.0 1.0 2.0 1000.0 VALUES 0.0 {} {} {}".format(
                            z_intermediate,
                            z_final,
                            z_final,
                        )
                    )
                    
                    mesh.add(displacement_x)
                    mesh.add(displacement_y)
                    mesh.add(displacement_z)
                    
                    # Apply Dirichlet boundary condition with displacement functions
                    # The nodes at the bottom are also fixed in all 3 rotational DOFs
                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 "
                            "VAL 1 1 1 0 0 0 0 0 0 "
                            "FUNCT {} {} {} 0 0 0 0 0 0",
                            format_replacement=[displacement_x, displacement_y, displacement_z],
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                
                # Nodes in compressed region: Radial compression only
                elif node.coordinates[2] <= threshold:
                    node_set = GeometrySet(node)
                    
                    displacement = calculate_displacement_for_cylinder(
                            node.coordinates, 
                            new_radius
                        )
                    # Create radial displacement functions (x,y only)
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
                    
                    # Apply radial compression (x,y constrained, z free, rotation free)
                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            "NUMDOF 9 ONOFF 1 1 0 0 0 0 0 0 0 "
                            "VAL 1 1 0 0 0 0 0 0 0 "
                            "FUNCT {} {} 0 0 0 0 0 0 0",
                            format_replacement=[displacement_x, displacement_y],
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
    
        
        # Add beam-to-solid surface contact for second simulation step
        if second_simulation:
            mesh.add(
                BoundaryCondition(
                    beam_lines,
                    "COUPLING_ID 2",
                    bc_type=mpy.bc.beam_to_solid_surface_contact,
                )
            )



    yz_ratio, degrees = calculate_angle_for_intersections(n_intersections, interval, cylinder_radius)

    interval[1] = interval[1] * yz_ratio
    
    # Generate the complete contour device beam structure
    print(f"Creating contour device with {number_of_wires} wires...")
    create_multiple_beams_shifted_in_y(
        mesh,
        number_of_wires,
        cylinder_radius,
        interval,
        n_el,
        True,
        mat,
        beam_object,
        compression_factor,
        compressed_part
    )
    
    # Generate VTK output for visualization
    mesh.write_vtk("contour_device_beams", base_dir)
    
    # Create 4C input file
    input_file = InputFile(cubit=cubit)
    input_file.add(mesh)
    
    # Generate simulation parameters
    input_file.add(
        f"""
        ------------------------------------------------------------------TITLE
        Medical Contour Device Simulation - Author: Mert Kayabek
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
        """,
        option_overwrite=True,
    )
    
    return input_file

preview = False


def create_straight_toy_aneurysm(cubit, config, restart):
    """
    Create a parametric straight cylindrical aneurysm geometry using Cubit.
    
    This function generates a simplified aneurysm model consisting of a
    cylindrical artery with an aneurysmal bulge. The geometry is created
    using Cubit's CAD capabilities and meshed with hexahedral elements.
    
    Args:
        cubit: CubitPy interface object for geometry creation
        config (dict): Aneurysm configuration parameters
        restart (bool): Whether to add contact surfaces for beam interaction
    """
    # Extract aneurysm geometry parameters
    L = config["L"]                    # Total artery length
    r_art_i = config["r_art_i"]        # Artery inner radius
    t = config["t"]                    # Wall thickness
    r_ca_i = config["r_ca_i"]          # Aneurysm inner radius
    h_y_ca = config["h_y_ca"]          # Aneurysm height offset
    n_ref_surf = config["n_ref_surf"]  # Surface refinement level
    n_ref_neck = config["n_ref_neck"]  # Neck refinement level
    
    print(f"Creating aneurysm: artery radius={r_art_i}, aneurysm radius={r_ca_i}, length={L}")
    
    # Validate geometry parameters
    if r_art_i + 2 * t + r_ca_i < h_y_ca:
        print("Warning: The specified height h_y_ca is too big. Cannot create the geometry.")

    # Create inner artery cylinder
    cylinder_i = cubit.cylinder(L, r_art_i, r_art_i, r_art_i)

    # Create aneurysm sphere
    sphere_i = cubit.sphere(r_ca_i)
    lumen = cubit.get_last_id("volume")

    # Create group for lumen tracking
    group_id = cubit.create_new_group()
    cubit.add_entity_to_group(group_id, lumen, "volume")
    
    # Position aneurysm sphere and unite with cylinder
    cubit.cmd(f"volume {sphere_i.id()} move 0 {h_y_ca} 0")
    cubit.cmd(f"unite {sphere_i.id()} {cylinder_i.id()}")

    # Get united volume and apply positioning
    united_volumes = cubit.get_last_id("volume")
    cubit.cmd(f"move volume {united_volumes} Y -2.5 Z 0.0")
    cubit.cmd(f"Rotate Volume {united_volumes} about X Angle 90")

    # Apply geometric tweaks for smooth transitions
    curves_to_tweak = [3]  # Aneurysm neck curve
    for curve_id in curves_to_tweak:
        cubit.cmd(f"Tweak Curve {curve_id} Fillet Radius {2*t}")
    
    # Generate mesh on main surfaces
    print("surf_ids:", cubit.get_group_surfaces(group_id))
    cubit.cmd(f"surface {5} {6} {7} Scheme Auto")
    cubit.cmd("surface {5} {6} {7} size auto factor 4")
    cubit.cmd(f"mesh surface {5} {6} {7}")

    # Apply surface refinement
    for n in range(int(n_ref_surf)):
        cubit.cmd(f"refine surface {5} {6} {7}")

    outer_surfaces = []
    cubit.cmd(f"merge surface {5} {6} {7}")

    for surf in cubit.volume(cubit.get_last_id(cupy.geometry.volume)).surfaces():
        if surf.id() in [5, 6, 7]:
            outer_surfaces.append(surf)

    # Extrude mesh to create wall thickness
    extrude_mesh_normal_to_surface(
        cubit, outer_surfaces, t, n_layer=1, average_normals=1
    )

    wall_id = 1

    # Add material block for arterial wall
    cubit.add_element_type(
        cubit.group(add_value=f"add volume {wall_id}"),
        cupy.element_type.hex8,
        name="arterial wall",
        material="MAT 3",
        bc_description="KINEM nonlinear",
    )

    # Apply boundary conditions for beam-surface contact
    for surf in cubit.volume(wall_id).surfaces():
        inner_surf = 4  # Inner surface ID for contact
        if surf.id()==inner_surf and restart:
            cubit.add_node_set(
                cubit.group(add_value=f"add surface {surf.id()}"),
                name="CONTACT SURFACE 2 - ART ",
                bc_section="BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE",
                bc_description="COUPLING_ID 2",
            )

    # Fix outer surface to prevent rigid body motion
    outer_ca_surf = 9  # Outer surface ID
    cubit.add_node_set(
        cubit.group(add_value=f"add surface {outer_ca_surf}"),
        name="fix axial direction artery {}".format(outer_ca_surf),
        bc_section="DESIGN SURF DIRICH CONDITIONS",
        bc_description="NUMDOF 6 ONOFF 1 1 1 0 0 0  VAL 0 0 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0 0 0",
    )
    
    print("Aneurysm geometry and mesh created successfully")


def setup_simulation(config, second_simulation):
    """
    Set up complete contour device-aneurysm simulation.
    
    This function orchestrates the creation of both the aneurysm geometry
    and the contour device, combining them into a single simulation.
    
    Args:
        config (dict): Complete configuration dictionary
        second_simulation (bool): Whether this is the second simulation step
        
    Returns:
        InputFile: Complete 4C input file for the simulation
    """
    print(f"Setting up {'second' if second_simulation else 'first'} simulation step...")
    
    # Initialize Cubit interface
    cubit = CubitPy()
    
    # Create aneurysm geometry
    create_straight_toy_aneurysm(cubit, config["aneurysm"], second_simulation)
    
    # Display geometry if preview is enabled
    if preview:
        cubit.display_in_cubit()
    
    # Create contour device beam structure
    input_file = create_beams_wrapped_around_cylinder(
        cubit,
        second_simulation,
        config=config,
    )
    
    # Add material properties for solid elements
    mat_mc = MaterialStVenantKirchhoff(youngs_modulus=116000.0, nu=0.3, density=4506000)
    input_file.add(mat_mc)
    
    mat_artery = MaterialStVenantKirchhoff(youngs_modulus=116000.0, nu=0.3, density=4506000)
    input_file.add(mat_artery)
    print("Simulation setup completed successfully")
    return input_file

# Global simulation settings
preview = False


if __name__ == "__main__":
    """
    Main execution section for contour device-aneurysm simulation.
    
    This script runs a two-step simulation:
    1. Contour device undeployed shape simulation
    2. Contour device-aneurysm contact simulation
    """
    
    print("=" * 80)
    print("Medical Contour Device Simulation")
    print("Author: Mert Kayabek")
    print("=" * 80)
    
    # Load configuration from YAML file
    pwd = os.getcwd()
    config_path = "/home_student/kayabek/sw/meshpy/ContourProject/Virtual_Deployment/config.yml"
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    print(f"Configuration loaded from: {config_path}")
    
    # ========================================================================
    # STEP 1: Contour Device Deployment Simulation
    # ========================================================================
    
    print("\\n" + "="*60)
    print("STEP 1: Contour Device Deployment Simulation")
    print("="*60)
    
    # Get base output directory from config and create simulation step directories
    base_output_dir = config["contour"]["output_directory"]
    simulation_dir_1 = os.path.join(base_output_dir, "simulation_step_1")
    os.makedirs(simulation_dir_1, exist_ok=True)
    
    # Generate first simulation input
    inputfile_1 = setup_simulation(config, False)
    
    # Clean simulation directory
    clean_simulation_directory(simulation_dir_1)
    
    # Write and run first simulation
    fd_placement_dat = os.path.join(simulation_dir_1, "fd-vmc-art.dat")
    inputfile_1.write_input_file(fd_placement_dat)
    
    print(f"Running shape generation simulation...")
    print(f"Input file: {fd_placement_dat}")
    print(f"Output directory: {simulation_dir_1}")
    
    run_four_c(fd_placement_dat, simulation_dir_1)
    
    # ========================================================================
    # STEP 2: Contour Device-Aneurysm Contact Simulation
    # ========================================================================
    print("\\n" + "="*60)
    print("STEP 2: Contour Device-Aneurysm Contact Simulation")
    print("="*60)
    
    # Use same base output directory from config for simulation step 2
    simulation_dir_2 = os.path.join(base_output_dir, "simulation_step_2")
    
    os.makedirs(simulation_dir_2, exist_ok=True)
    
    # Clean simulation directory
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
                    GEOMETRY_PAIR_SEGMENTATION_SEARCH_POINTS 6
                    GAUSS_POINTS                             6
                    GEOMETRY_PAIR_STRATEGY                   segmentation
                    PENALTY_LAW                              linear_quadratic
                    PENALTY_PARAMETER                        100.0
                    PENALTY_PARAMETER_G0                     0.0001
                    MORTAR_SHAPE_FUNCTION                    line2
                    MORTAR_CONTACT_DEFINED_IN                reference_configuration
                """,
                option_overwrite=True,
    )
    
    inputfile_1.write_input_file(fd_placement_dat2)
    
    print(f"Running contact simulation...")
    print(f"Input file: {fd_placement_dat2}")
    print(f"Output directory: {simulation_dir_2}")
    
    run_four_c(
        fd_placement_dat2,
        simulation_dir_2,
        restart_step=config["time"]["steps1"],
        restart_from=os.path.join(os.path.relpath(simulation_dir_1, simulation_dir_2), "xxx"),
    )