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
Arc-Based Contour Device Parameter Sweep Simulation

This script performs comprehensive parameter sweep studies for arc-based medical 
contour devices using parametric curved beam structures. The simulation includes:

1. Arc-based contour device beam structure generation with elliptical geometry
2. Systematic parameter variation across multiple design dimensions
3. Automated result processing and visualization generation
4. Advanced retry mechanisms for simulation robustness
5. Configuration-driven parameter management using YAML files

Key Features:
- Parametric arc geometry generation using elliptical parametric curves
- Arc length-based node positioning for precise intersection calculations
- Comprehensive parameter sweep capabilities with configurable ranges
- Automatic beam intersection detection and penalty-based coupling
- Memory-efficient result processing with selective file retention
- Retry logic with adaptive time stepping for failed simulations
- VTK output generation for visualization and analysis

The simulation supports both single parameter combinations and comprehensive sweeps for design optimization and sensitivity analysis.

Author: Mert Kayabek
"""

# Standard library imports
import numpy as np
import autograd.numpy as npAD
import os
import itertools
import glob
import time
import sys
import shutil
import re
from datetime import datetime, timedelta

# Third-party imports
import pyvista as pv
import yaml
#pv.start_xvfb()  # Start virtual X server for headless rendering

# MeshPy core modules
from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.core.mesh import Mesh
from meshpy.core.rotation import Rotation

# MeshPy 4C interface modules
from meshpy.four_c.boundary_condition import BoundaryCondition
from meshpy.four_c.element_beam import Beam3rHerm2Line3
from meshpy.four_c.function import Function
from meshpy.four_c.input_file import InputFile
from meshpy.four_c.material import MaterialReissner
from meshpy.four_c.run_four_c import run_four_c

# MeshPy utility modules
from meshpy.utils.nodes import get_single_node, find_close_nodes

# MeshPy mesh creation modules
from meshpy.mesh_creation_functions.beam_basic_geometry import (
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_line,
)
from meshpy.mesh_creation_functions.beam_curve import create_beam_mesh_curve


def create_beams_wrapped_around_cylinder(
    base_dir, 
    interval_end,
    cylinder_radius,
    compressed_part,
    beam_radius,
    positional_coupling_penalty,
    rotational_coupling_penalty,
    z_scale_factor=1.0,
    number_of_wires=12,
    youngs_modulus=83000,
    preview=False
):
    """
    Create an arc-based contour device beam structure wrapped around a cylindrical surface.
    
    This function generates multiple curved beams that form a medical contour device using
    parametric elliptical arc geometry. The beams are arranged to intersect at calculated
    points and provide specific coverage patterns for medical treatment applications.
    
    The key difference from line-based devices is that this uses curved (arc) geometry
    with sophisticated arc length calculations to ensure proper node placement and
    intersection patterns.
    
    Args:
        base_dir (str): Output directory for simulation files and results
        interval_end (float): End z-coordinate for beam generation interval
        cylinder_radius (float): Radius of the cylindrical surface for beam wrapping
        compressed_part (float): Fraction of beam length to apply compression to
        beam_radius (float): Cross-sectional radius of individual beam elements
        positional_coupling_penalty (float): Penalty parameter for positional coupling
        rotational_coupling_penalty (float): Penalty parameter for rotational coupling
        z_scale_factor (float): Scaling factor for z-direction arc height (default: 1.0)
        number_of_wires (int): Number of arc beams to create (default: 12)
        youngs_modulus (float): Young's modulus of beam material in N/mm² (default: 83000)
        preview (bool): Whether to show preview visualization (default: False)
        
    Returns:
        InputFile: Complete 4C input file for arc-based contour device simulation
        
    Arc Geometry Features:
        - Elliptical arc shapes in cylindrical coordinates
        - Arc length-based node positioning for uniform distribution
        - Parametric curve generation with configurable scaling
        - Intersection point calculation based on geometric analysis
    """
    
    # Define simulation parameters
    interval = [0, interval_end]  # z-coordinate range for beam generation
    n_intersections = 2  # Minimum number of beam intersections for stability
    compression_factor = 0.95  # Maximum compression ratio (5% reduction)

    # Calculate total number of elements based on intersection requirements
    n_el = 8 * (n_intersections - 1) * number_of_wires
    
    # Time stepping parameters for static analysis
    num_steps = 20  # Number of load steps for gradual application
    time_step = 1 / num_steps  # Individual time step size
    
    # Initialize mesh and material objects
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

    def create_multiple_beams_shifted_in_y(
        mesh,
        number_of_wires,
        cylinder_radius,
        interval,
        n_el,
        add_sets,
        material,
        beam_object,
        compression_factor,
        compressed_part,
    ):
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

        for node in mesh.nodes:
            if not node.is_middle_node:
                
                # Nodes at the bottom
                if  np.linalg.norm(node.coordinates[2] - interval[0]) < 1e-9:                
                    node_set = GeometrySet(node)
                    
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

                    # Apply Dirichlet boundary condition with displacement functions
                    # The nodes at the bottom are also fixed in all 3 rotational DOFs
                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 "
                                "VAL 1 1 0 0 0 0 0 0 0 "
                                "FUNCT {} {} 0 0 0 0 0 0 0" 
                            ),
                            format_replacement=[displacement_x, displacement_y],
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

    yz_ratio, degrees = calculate_angle_for_intersections(n_intersections, interval, cylinder_radius)
    
    interval[1] = interval[1] * yz_ratio

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
        RESULTSEVERY                           1
        NLNSOL                                fullnewton
        DIVERCONT                             adapt_step
        TIMESTEP                              {time_step}
        NUMSTEP                               {num_steps}
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
        ELEMENT_GID                           yes
        ----------------------------------------------------------------BINNING STRATEGY
        BIN_SIZE_LOWER_BOUND                  3.0
        DOMAINBOUNDINGBOX                     -30 -30 -30 30 30 30
        ----------------------------------------------------------------BEAM INTERACTION
        REPARTITIONSTRATEGY                   Everydt
        SEARCH_STRATEGY                       bounding_volume_hierarchy
        """
    )

    return input_file

def visualize_timestep(vtk_file_path, output_path):
    """
    Creates a PyVista visualization of a VTK file and saves it as an image.
    
    Args:
        vtk_file_path (str): Absolute path to the VTK file to visualize
        output_path (str): Absolute path for the output image file
        
    Returns:
        bool: True if visualization was created successfully, False otherwise
        
    Error Handling:
        - Checks file existence before processing
        - Handles PyVista rendering errors
        - Reports specific error messages for debugging
        
    Usage in Parameter Sweeps:
        This function is typically called for key timesteps (0, middle, final)
        to create visualization summaries of simulation results without storing
        the complete time history.
    """
    # Validate input file existence
    if not os.path.exists(vtk_file_path):
        print(f"Warning: VTK file not found: {vtk_file_path}")
        return False
    
    try:
        # Load the VTK file
        mesh = pv.read(vtk_file_path)
        
        # Create a plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh, color='tan', show_edges=True, line_width=1)
        
        # Add a title
        plotter.add_text(os.path.basename(vtk_file_path), position='upper_edge')
        
        # Set the view
        plotter.view_isometric()
        plotter.camera.zoom(1.2)
        
        # Save to image file
        plotter.screenshot(output_path)
        plotter.close()
        
        print(f"Created visualization: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return False
    
def copy_vtk_files(results_dir, viz_dir, keep_timesteps=[0, 25, 50]):
    """
    Copy VTK simulation result files for specified timesteps to visualization directory.
    
    This function implements memory-efficient result processing by selectively copying
    only the most important simulation timesteps rather than storing all output files.
    It handles both parallel VTK master files (.pvtu) and their associated piece files
    (.vtu) to ensure complete visualization datasets.
    
    The function is designed for parameter sweep workflows where disk space is critical
    and only key simulation states need to be preserved for analysis.
    
    File Types Processed:
        - .pvtu files: Parallel VTK master files containing metadata
        - .vtu files: Individual processor piece files with actual data
        - .log/.err files: Simulation logs and error messages
    
    Args:
        results_dir (str): Source directory containing simulation results
        viz_dir (str): Destination directory for copied visualization files
        keep_timesteps (list): List of timestep indices to preserve (default: [0, 25, 50])
        
    Returns:
        bool: True if files were successfully copied, False if no files found
        
    File Naming Convention:
        The function expects 4C simulation output files with naming pattern:
        - structure-beams-{timestep:05d}.pvtu (master files)
        - structure-beams-{timestep:05d}-{proc}.vtu (piece files)
        
    Error Handling:
        - Gracefully handles missing VTK directories
        - Reports missing timesteps without failing
        - Continues processing if individual files are missing
        
    Memory Optimization:
        By copying only selected timesteps (typically beginning, middle, end),
        this function reduces storage requirements by 90%+ while preserving
        essential simulation data for visualization and analysis.
    """
    # Locate VTK files directory within simulation results
    vtk_files_dir = os.path.join(results_dir, "xxx-vtk-files")
    if not os.path.exists(vtk_files_dir):
        print(f"Warning: VTK files directory not found: {vtk_files_dir}")
        return False
    
    files_copied = 0
    
    # Process each specified timestep
    for timestep in keep_timesteps:
        # Format the timestep to match file pattern
        timestep_str = f"{timestep:05d}"
        
        # First find all pvtu files for this timestep (master files)
        pvtu_files = glob.glob(os.path.join(vtk_files_dir, f"structure-beams-{timestep_str}.pvtu"))
        
        # Find all associated piece files for this timestep
        vtu_files = glob.glob(os.path.join(vtk_files_dir, f"structure-beams-{timestep_str}-*.vtu"))
        
        # Combine all files for this timestep
        all_files = pvtu_files + vtu_files
        
        if all_files:
            # Copy all files for this timestep
            for src_file in all_files:
                dest_file = os.path.join(viz_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
                files_copied += 1
        else:
            print(f"No VTK files found for timestep {timestep}")

    # Copy log and error files from results directory
    log_files = glob.glob(os.path.join(results_dir, "xxx.log"))
    err_files = glob.glob(os.path.join(results_dir, "xxx.err"))
    
    for src_file in log_files + err_files:
        if os.path.exists(src_file):
            dest_file = os.path.join(viz_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dest_file)
            files_copied += 1
    
    return files_copied > 0


def parameter_sweep():
    """
    Execute comprehensive parameter sweep for arc-based contour device simulations.
    
    This function performs systematic parameter studies by varying multiple design
    and material properties of the arc-based contour device. It creates a complete
    simulation matrix exploring different combinations of geometric, material, and
    coupling parameters to understand device behavior and optimize performance.
    
    The function implements the following workflow:
    1. Load configuration parameters from YAML file
    2. Generate parameter combination matrix
    3. Check for existing results to avoid duplication
    4. Execute simulations with retry logic for failed cases
    5. Process and store results with memory optimization
    6. Generate comprehensive sweep summary
    
    Parameter Sweep Dimensions:
        - Material Properties: Young's modulus variations for different wire materials
        - Geometric Properties: beam radius, compression ratios, z-scaling factors
        - Interaction Parameters: positional and rotational coupling penalties
        - Device Configuration: derived from base geometry parameters
    
    Advanced Features:
        - Automatic retry with adaptive time stepping for convergence issues
        - Memory-efficient result processing with selective file retention
        - Dynamic timestep determination based on simulation output
        - Progress tracking with estimated completion times
        - Error handling and logging for failed simulations
        - Duplicate detection to resume interrupted sweeps
    
    Configuration Loading:
        All parameters are loaded from config.yml including:
        - Simulation directories (output and existing results)
        - Geometric parameters (interval_end, cylinder_radius, number_of_wires)
        - Parameter arrays for systematic variation
    
    Output Structure:
        Results are organized in timestamped directories with parameter-specific
        subdirectories containing:
        - VTK files for key timesteps
        - Simulation logs and error files
        - Parameter configuration records
        - Visualization outputs (when enabled)
    
    Retry Logic:
        Failed simulations are automatically retried with progressively finer
        time stepping (increasing from 20 to 180 steps) to improve convergence.
        
    """
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Extract directory configurations from config file
    base_output_dir = config.get("output_directory", "/home_student/kayabek/sw/Results")
    existing_sweeps_dir = config.get("existing_sweeps_directory", "/home_student/kayabek/sw/Results/Sweeps_new")
    
    # Extract simulation parameters from config file
    interval_end = config["interval_end"]
    cylinder_radius = config["cylinder_radius"]
    number_of_wires = config["number_of_wires"]
    youngs_moduli = np.array(config["youngs_moduli"])  # Different material stiffness values (N/mm²)
    beam_radii = np.array(config["beam_radii"])  # Different beam cross-sectional radii (mm)
    compressed_parts = np.array(config["compressed_parts"])  # Compression ratios (0 = no compression)
    positional_penalties = np.array(config["positional_penalties"])  # Positional coupling penalties
    rotational_penalties = np.array(config["rotational_penalties"])  # Rotational penalties
    z_scale_factors = np.array(config["z_scale_factors"])  # Z-direction arc scaling factors

    # Default timesteps for result processing (beginning, middle, end)
    keep_timesteps = [0, 25, 50]

    # Timing variables
    sweep_start_time = time.time()
    
    # Create timestamped base directory for this parameter sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(base_output_dir, f"parameter_sweep_{timestamp}")
    os.makedirs(base_results_dir, exist_ok=True)

    # Save parameter configuration for reference
    with open(os.path.join(base_results_dir, "parameters.txt"), "w") as f:
        f.write(f"Parameter sweep configuration:\n")
        f.write(f"interval_end: {interval_end}\n")
        f.write(f"cylinder_radius: {cylinder_radius}\n")
        f.write(f"youngs_moduli: {youngs_moduli}\n")
        f.write(f"beam_radii: {beam_radii}\n")
        f.write(f"compressed_parts: {compressed_parts}\n")
        f.write(f"positional_penalties: {positional_penalties}\n")
        f.write(f"rotational_penalties: {rotational_penalties}\n")
        f.write(f"z_scale_factors: {z_scale_factors}\n")

    # Loop through parameter combinations
    total_combinations = len(youngs_moduli) * len(beam_radii) * len(compressed_parts) * len(positional_penalties) * len(rotational_penalties) * len(z_scale_factors)
    print(f"Total parameter combinations: {total_combinations}")
    
    current_combination = 0
    skipped_combinations = 0
    
    # Loop through parameter combinations
    for youngs_modulus, beam_radius in itertools.product(youngs_moduli, beam_radii):
        for compressed_part, z_scale_factor in itertools.product(compressed_parts, z_scale_factors):
            for pos_penalty, rot_penalty in itertools.product(positional_penalties, rotational_penalties):
                current_combination += 1
                combination_start_time = time.time()
                
                # Create parameter directory name
                params_dir_name = (
                    f"nbeams{number_of_wires}_"
                    f"int{interval_end:.1f}_"
                    f"rad{cylinder_radius:.2f}_"
                    f"comp{compressed_part:.2f}_"
                    f"beam{beam_radius}_"
                    f"zscale{z_scale_factor:.2f}_"
                    f"young{youngs_modulus}_"
                    f"pos{pos_penalty}_"
                    f"rot{rot_penalty}"
                )
                # Check if this combination already exists in the Sweeps directory
                existing_dir = os.path.join(existing_sweeps_dir, params_dir_name)
                
                if os.path.exists(existing_dir):
                    print(f"\nSkipping combination {current_combination}/{total_combinations}: Already exists")
                    print(f"Parameters: young={youngs_modulus}, beam_radius={beam_radius}, "
                          f"compressed_part={compressed_part}, zscale={z_scale_factor}, "
                          f"pos_penalty={pos_penalty}, rot_penalty={rot_penalty}")
                    skipped_combinations += 1
                    continue
                
                # Start timing for this combination
                combination_start_time = time.time()

                params_dir = os.path.join(base_results_dir, params_dir_name)
                os.makedirs(params_dir, exist_ok=True)
                
                results_dir = os.path.join(params_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                
                # Log progress
                print(f"\nRunning combination {current_combination}/{total_combinations} ({(current_combination/total_combinations)*100:.1f}%)")
                print(f"Parameters: young={youngs_modulus}, beam_radius={beam_radius}, "
                      f"compressed_part={compressed_part}, zscale={z_scale_factor}, "
                      f"pos_penalty={pos_penalty}, rot_penalty={rot_penalty}")
                
                try:
                    # Create and run simulation
                    input_file = create_beams_wrapped_around_cylinder(
                        params_dir,
                        interval_end,
                        cylinder_radius,
                        compressed_part,
                        beam_radius,
                        pos_penalty,
                        rot_penalty,
                        z_scale_factor,
                        number_of_wires,
                        youngs_modulus=youngs_modulus
                    )
                    
                    input_file_path = os.path.join(params_dir, "contour_device_beams.dat")
                    input_file.write_input_file(input_file_path)
                    
                    # Run simulation
                    #print(f"Running simulation...")
                    return_code = run_four_c(
                        input_file_path,
                        results_dir,
                        output_name='xxx',
                        n_proc=2
                    )
                    combination_end_time = time.time()
                    time_for_combination = combination_end_time - combination_start_time
                    time_str = str(timedelta(seconds=int(time_for_combination)))
                    print(f"  Time for this combination: {time_str}")

                    # Process results
                    if return_code == 0:
                        print(f"Simulation completed successfully.")
                        
                        # Create visualization directory
                        viz_dir = os.path.join(params_dir, "visualizations")
                        os.makedirs(viz_dir, exist_ok=True)
                        
                        # Create visualizations for important timesteps
                        vtk_dir = os.path.join(results_dir, "xxx-vtk-files")
                        visualization_created = False

                        # Dynamically determine timesteps based on number of files
                        if os.path.exists(vtk_dir):
                            # Count the number of files in the vtk directory
                            file_count = len(os.listdir(vtk_dir))
                            # Calculate the last timestep (file_count / 6 - 1)
                            last_timestep = int(file_count / 6) - 1
                            # Calculate the middle timestep ((0 + last_timestep) / 2)
                            middle_timestep = (last_timestep + 1) // 2
                            
                            # Update keep_timesteps with dynamically calculated values
                            keep_timesteps = [0, middle_timestep, last_timestep]
                            print(f"Dynamically determined timesteps: {keep_timesteps}")
                        else:
                            print(f"VTK directory not found, using default timesteps: {keep_timesteps}")

                        visualization_created = copy_vtk_files(results_dir, viz_dir, keep_timesteps)
                        """
                        for timestep in keep_timesteps:
                            # Format the timestep to match file pattern
                            timestep_str = f"{timestep:05d}"
                            
                            # Look for structure files
                            structure_files = glob.glob(os.path.join(vtk_dir, f"structure-beams-{timestep_str}*.v*u"))
                            
                            if structure_files:
                                # Use the first matching file
                                vtk_file = structure_files[0]
                                viz_file = os.path.join(viz_dir, f"timestep_{timestep}.png")
                                visualization_created = copy_vtk_files(results_dir, viz_dir, keep_timesteps)
                                # Create the visualization

                                #if visualize_timestep(vtk_file, viz_file):
                                #    visualization_created = True

                        """
                        # Delete results directory to save space
                        if visualization_created:
                            try:
                                shutil.rmtree(results_dir)
                                #print(f"Deleted results directory to save space")
                            except Exception as e:
                                print(f"Error deleting results directory: {str(e)}")
                        else:
                            print(f"Warning: No visualizations were created, keeping results directory")
                    else:
                        print(f"Simulation failed with return code {return_code}")
                        # Try with progressively finer time stepping
                        original_num_steps = 20  # Starting value
                        max_num_steps = 180  # Max value to try
                        current_num_steps = original_num_steps + 20  # First retry with 40 steps
                        
                        # Keep retrying with increased time step resolution until we succeed or reach max
                        while current_num_steps <= max_num_steps:
                            print(f"Retrying with finer time stepping: num_steps={current_num_steps}")
                            retry_start_time = time.time()
                            
                            # Calculate new time step
                            new_time_step = 1.0 / current_num_steps
                            """
                            # Create a modified input file with new time parameters
                            retry_input_file = create_beams_wrapped_around_cylinder(
                                params_dir,
                                interval_end,
                                cylinder_radius,
                                compressed_part,
                                beam_radius,
                                pos_penalty,
                                rot_penalty,
                                z_scale_factor,
                                number_of_wires,
                                youngs_modulus=youngs_modulus
                            )
                            """
                            # Modify the time step parameters
                            retry_input_file_content = input_file.get_string()
                            retry_input_file_content = retry_input_file_content.replace(
                                f"TIMESTEP                              {1/original_num_steps}", 
                                f"TIMESTEP                              {new_time_step}"
                            ).replace(
                                f"NUMSTEP                               {original_num_steps}",
                                f"NUMSTEP                               {current_num_steps}"
                            )
                            
                            # Write the modified input file
                            retry_input_file_path = os.path.join(params_dir, f"contour_device_beams.dat")
                            with open(retry_input_file_path, "w") as f:
                                f.write(retry_input_file_content)
                            
                            # Clean up previous results
                            if os.path.exists(results_dir):
                                shutil.rmtree(results_dir)
                            os.makedirs(results_dir, exist_ok=True)
                            
                            # Run the retry simulation
                            retry_return_code = run_four_c(
                                retry_input_file_path,
                                results_dir,
                                output_name='xxx',
                                n_proc=2
                            )
                            
                            # Check if retry was successful
                            if retry_return_code == 0:
                                print(f"Retry successful with num_steps={current_num_steps}")
                                
                                # Process successful results
                                viz_dir = os.path.join(params_dir, "visualizations")
                                os.makedirs(viz_dir, exist_ok=True)
                                
                                vtk_dir = os.path.join(results_dir, "xxx-vtk-files")
                                visualization_created = False
                                
                                if os.path.exists(vtk_dir):
                                    # Determine timesteps dynamically
                                    file_count = len(os.listdir(vtk_dir))
                                    last_timestep = int(file_count / 6) - 1
                                    middle_timestep = (last_timestep + 1) // 2
                                    keep_timesteps = [0, middle_timestep, last_timestep]
                                    print(f"Using timesteps: {keep_timesteps}")
                                    
                                    visualization_created = copy_vtk_files(results_dir, viz_dir, keep_timesteps)
                                
                                # Clean up after successful retry
                                if visualization_created:
                                    try:
                                        shutil.rmtree(results_dir)
                                    except Exception as e:
                                        print(f"Error deleting results directory: {str(e)}")
                                
                                # Exit retry loop on success
                                break
                            else:
                                print(f"Retry failed with num_steps={current_num_steps}, return code {retry_return_code}")
                                # Continue to next retry attempt with more time steps
                                retry_time = time.time() - retry_start_time
                                print(f"Retry took {str(timedelta(seconds=int(retry_time)))}")
                                
                                # Try to save any output from this failed attempt
                                viz_dir = os.path.join(params_dir, f"visualizations_attempt_{current_num_steps}")
                                os.makedirs(viz_dir, exist_ok=True)
                                
                                vtk_dir = os.path.join(results_dir, "xxx-vtk-files")
                                if os.path.exists(vtk_dir):
                                    copy_vtk_files(results_dir, viz_dir, keep_timesteps)
                                
                                # Increment for next attempt
                                current_num_steps += 20
                        
                        # If we reached max steps without success, do final cleanup
                        if current_num_steps > max_num_steps:
                            print(f"All retry attempts failed, max num_steps={max_num_steps} reached")
                            
                            # Final attempt to copy any VTK files from the last attempt
                            viz_dir = os.path.join(params_dir, "visualizations")
                            os.makedirs(viz_dir, exist_ok=True)
                            
                            vtk_dir = os.path.join(results_dir, "xxx-vtk-files")
                            if os.path.exists(vtk_dir):
                                copy_vtk_files(results_dir, viz_dir, keep_timesteps)
                            
                            # Save log file with information about the failure
                            with open(os.path.join(params_dir, "retry_failed.log"), "w") as f:
                                f.write(f"All retry attempts failed, tried up to num_steps={max_num_steps}\n")
                            
                            # Clean up results directory
                            try:
                                shutil.rmtree(results_dir)
                            except Exception as e:
                                print(f"Error deleting results directory: {str(e)}")
                    # Keep the results directory for debugging    
                except Exception as e:
                    # Log errors but continue with next parameter combination
                    print(f"Error in simulation: {str(e)}")
                    with open(os.path.join(params_dir, "error.log"), "w") as f:
                        f.write(f"Error: {str(e)}\n")
                    continue
    # Print summary
    print(f"\nParameter sweep completed:")
    print(f"Total combinations: {total_combinations}")
    print(f"Skipped (already existed): {skipped_combinations}")
    print(f"Processed: {total_combinations - skipped_combinations}")
    total_time = time.time() - sweep_start_time
    print(f"Total time: {str(timedelta(seconds=int(total_time)))}")

if __name__ == "__main__":
    """Execution part of script."""
    parameter_sweep()
