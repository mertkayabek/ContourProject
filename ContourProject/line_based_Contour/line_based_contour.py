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
Line-Based Contour Device Simulation

This script creates a comprehensive simulation of a line-based medical contour device
using parametric beam structures. The simulation includes:
1. Line-based contour device beam structure generation
2. Cylindrical wrapping geometry for medical applications
3. Beam-to-beam interaction simulation with coupling
4. Configuration-driven parameter management

The contour device uses multiple helical beams arranged in a cylindrical pattern
to provide medical treatment coverage. The beams intersect at specific points
and are coupled together to simulate realistic device behavior.

The simulation is performed with configurable compression to simulate
geometry creation and device expansion.

Features:
- Fully config-driven workflow using YAML configuration
- Parametric beam geometry generation
- Automatic beam intersection detection and coupling
- Configurable material properties and simulation parameters
- VTK output for visualization
- 4C finite element simulation integration

Author: Mert Kayabek
"""

# Standard library imports
import numpy as np
import autograd.numpy as npAD
import os
import itertools
import glob
from datetime import datetime
import shutil
import pyvista as pv
import re
#pv.start_xvfb()  # Start virtual X server for headless rendering
import time
import sys
from datetime import timedelta
import yaml

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


def create_beams_wrapped_around_cylinder(config, preview=False):
    """
    Create a line-based contour device beam structure wrapped around a cylindrical surface.
    
    This function generates multiple helical beams that form a medical contour device.
    The beams are arranged to intersect and provide specific coverage patterns
    for medical treatment applications. The device simulates deployment through
    marker-based compression and subsequent expansion.
    
    Args:
        config (dict): Configuration dictionary containing all simulation parameters
                      including geometry, material, timing, and output settings
        preview (bool): Whether to show preview visualization (default: False)
        
    Returns:
        InputFile: Complete 4C input file for contour device simulation
        
    Configuration sections used:
        - contour: Device geometry and material parameters
        - time: Time stepping and simulation duration
        - simulation: Solver settings and analysis type
        - output: Visualization and file output settings
        - beam_interaction: Coupling and interaction parameters
    """
    # Extract configuration parameters from different sections
    contour_config = config["contour"]
    time_config = config["time"]
    sim_config = config["simulation"]
    output_config = config["output"]
    beam_interaction_config = config["beam_interaction"]
    
    # Extract geometric parameters from config
    base_dir = contour_config["output_directory"]
    interval = [contour_config["interval_start"], contour_config["interval_end"]]  # z coordinate range
    cylinder_radius = contour_config["cylinder_radius"]  # radius of wrapping cylinder
    n_intersections = contour_config["n_intersections"]  # minimum 2 for stable intersection
    number_of_wires = contour_config["number_of_wires"]  # each direction, normally 72 total
    compression_factor = contour_config["compression_factor"]  # Maximum compression (0.9-0.95)
    compressed_part = contour_config["compressed_part"]  # ratio of beam compressed from start
    
    # Extract material properties from config
    beam_radius = contour_config["beam_radius"]  # radius of individual beam elements
    youngs_modulus = contour_config["youngs_modulus"]  # N/mm^2 Young's modulus of beam material
    
    # Extract coupling parameters for beam interactions

    positional_coupling_penalty = contour_config["pos_penalty"]  # penalty for positional coupling
    rotational_coupling_penalty = contour_config["rot_penalty"]  # penalty for rotational coupling
    
    # Calculate simulation parameters
    n_el = 8 * (n_intersections - 1) * number_of_wires  # Total number of elements
    num_steps = time_config["steps1"]
    time_step = time_config["dt_1"] / num_steps

    # Initialize mesh and material objects
    mesh = Mesh()
    mat = MaterialReissner(youngs_modulus=youngs_modulus, radius=beam_radius)
    beam_object = Beam3rHerm2Line3

    def find_intersections_and_apply_coupling(mesh, beams, positional_coupling_penalty, rotational_coupling_penalty):
        """
        Find intersections between beams and apply coupling constraints.
        
        This function identifies nodes that are geometrically close to each other
        (intersection points between different beams) and applies penalty-based
        coupling to simulate physical connections between the beams.
        
        Args:
            mesh: The mesh object containing all beam nodes
            beams: List of beam dictionaries (not actively used in current implementation)
            positional_coupling_penalty: Penalty parameter for position coupling
            rotational_coupling_penalty: Penalty parameter for rotation coupling
            
        Returns:
            GeometrySet: Set of intersecting nodes, or None if no intersections found
            
        Note:
            Uses MeshPy's find_close_nodes utility to automatically detect
            geometrically coincident nodes within a tolerance.
        """
        # Use find_close_nodes to find intersecting nodes
        # Returns list of lists where each inner list contains nodes that are close to each other
        close_node_groups = find_close_nodes(mesh.nodes)
        print("\nFound close node groups:")

        # Create GeometrySet for intersecting nodes
        # Start with first group that has multiple nodes as initial geometry
        first_group = next((group for group in close_node_groups if len(group) > 1), None)
        if not first_group:
            return None
            
        intersecting_nodes = GeometrySet(first_group[0])
        
        # Apply coupling for each group of close nodes
        for node_group in close_node_groups:
            if len(node_group) > 1:  # Only process groups with multiple intersecting nodes
                # Add all nodes in the group to the intersecting nodes set
                for node in node_group:
                    intersecting_nodes.add(node)
                
                # Apply penalty-based coupling between pairs of nodes in the group
                mesh.couple_nodes(
                    nodes=[node_group[0], node_group[1]],
                    coupling_type=mpy.bc.point_coupling_penalty,
                    coupling_dof_type=f"POSITIONAL_PENALTY_PARAMETER {positional_coupling_penalty} ROTATIONAL_PENALTY_PARAMETER {rotational_coupling_penalty}"
                )
        
        return intersecting_nodes

    
    def calculate_displacement_for_cylinder(coordinates, new_radius):
        """
        Calculate radial displacement for cylindrical compression transformation.
        
        This function computes the displacement required to compress a point
        from its current radial position to a new radial position on a cylinder.
        Used to simulate marker-based compression of the contour device.
        
        Args:
            coordinates (array): [x, y, z] coordinates of the point
            new_radius (float): Target radius for compression
            
        Returns:
            array: [dx, dy, dz] displacement vector (dz = 0 for cylindrical compression)
            
        Note:
            Only affects x and y coordinates (radial compression),
            z-coordinate remains unchanged.
        """
        x = coordinates[0]
        y = coordinates[1]
        
        # Calculate the original radial distance from z-axis
        original_radius = np.sqrt(x**2 + y**2)
        
        # Calculate the scaling factor for radial compression
        scaling_factor = new_radius / original_radius
        
        # Apply radial scaling to x and y coordinates
        new_x = x * scaling_factor
        new_y = y * scaling_factor
        
        # Return displacement vector (no z-displacement for cylindrical compression)
        return np.array([new_x-x, new_y-y, 0])
    
    def calculate_angle_for_intersections(n_intersections, interval, radius):
        """
        Calculate required helical angle for desired number of beam intersections.
        
        This function determines the helical pitch angle needed to achieve
        a specific number of intersections between helical beams wrapped
        around a cylinder of given radius.
        
        Args:
            n_intersections (int): Desired number of intersections between beams
            interval (list): [start, end] z-coordinate range of beam
            radius (float): Cylinder radius for beam wrapping
            
        Returns:
            tuple: (tan_alpha, alpha_degrees) where:
                - tan_alpha: Tangent of the helical angle
                - alpha_degrees: Helical angle in degrees
                
        Note:
            The calculation ensures that beams following helical paths
            will intersect at the specified number of points along their length.
        """
        length = interval[1] - interval[0]
        # Calculate helical angle based on desired intersections and geometry
        # Formula derived from helical geometry: circumferential distance vs. axial distance
        tan_alpha = ((n_intersections - 1) * np.pi * radius) / length
        alpha_degrees = np.degrees(np.arctan(tan_alpha))
        return tan_alpha, alpha_degrees

    def create_multiple_beams_shifted_in_y(
        mesh,
        number_of_wires,
        cylinder_radius,
        interval,
        n_el,
        add_sets,
        material,
        beam_object,
        tan_alpha,
        compression_factor,
        compressed_part
    ):
        """
        Create multiple helical beams arranged circumferentially around a cylinder.
        
        This function generates pairs of counter-rotating helical beams that form
        the main structure of the line-based contour device. Each beam follows
        a helical path and is shifted circumferentially to create an even distribution.
        
        Args:
            mesh: Mesh object to add beams to
            number_of_wires: Number of beam pairs to create
            cylinder_radius: Radius of the cylinder around which beams wrap
            interval: [start, end] z-coordinate range for beam generation
            n_el: Number of elements per beam
            add_sets: Whether to add node sets for boundary conditions
            material: Material object for the beams
            beam_object: Beam element type to use
            tan_alpha: Tangent of helical angle
            compression_factor: Factor for radial compression (0-1)
            compressed_part: Fraction of beam length to apply compression to
            
        Note:
            Creates two beams per iteration - one with positive helical angle
            and one with negative helical angle to ensure intersection patterns.
            Each beam is circumferentially shifted by 2π/number_of_wires.
        """

        def n_shape_yz(t):
            """Define positive helical beam shape in cylindrical coordinates."""
            x = cylinder_radius
            y = tan_alpha * t  # Helical component
            z = 1.0 * t        # Axial component
            return npAD.array([x, y, z])
        
        def n_shape_yz2(t):
            """Define negative helical beam shape in cylindrical coordinates."""
            x = cylinder_radius
            y = - tan_alpha * t  # Counter-helical component
            z = 1.0 * t          # Axial component
            return npAD.array([x, y, z])

        # Storage for beam information
        beams = []
        beams_start = []
        beams_end = []

        # Create beam pairs with circumferential distribution
        for i in range(number_of_wires):
            # Calculate circumferential shift for even distribution
            shift_i = (2.0 * npAD.pi * cylinder_radius / number_of_wires) * i
            
            def shape_with_shift(t, shift=shift_i):
                """Positive helical beam with circumferential shift."""
                base = n_shape_yz(t)
                return npAD.array([base[0], base[1] + shift, base[2]])

            # Create first beam (positive helical direction)
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
                """Negative helical beam with circumferential shift."""
                base = n_shape_yz2(t)
                return npAD.array([base[0], base[1] + shift, base[2]])

            # Create second beam (negative helical direction)
            dir2 = create_beam_mesh_curve(
                mesh,
                beam_object,
                material,
                shape_with_shift2,
                interval=interval,
                n_el=n_el,
                add_sets=add_sets
            )
            
            # Store beam information for later processing
            beams.append(dir1)
            beams.append(dir2)
            beams_start.append(dir1["start"])
            beams_start.append(dir2["start"])
            beams_end.append(dir1["end"])
            beams_end.append(dir2["end"])

        # Apply cylindrical wrapping transformation to all beams
        mesh.wrap_around_cylinder(radius=cylinder_radius)
        
        # Find intersection points and apply coupling between beams
        find_intersections_and_apply_coupling(mesh, beams, positional_coupling_penalty, rotational_coupling_penalty)
        
        mpy.check_overlapping_elements = False
        
        # Calculate compressed radius for compressive simulation
        new_radius = cylinder_radius * (1.0 - compression_factor)
        
        # Apply boundary conditions to simulate compression
        for node in mesh.nodes:
            if not node.is_middle_node:  # Only apply to end nodes
                
                # Identify start nodes (at z = interval[0]) and apply fixed constraints
                if np.linalg.norm(node.coordinates[2] - interval[0]) < 1e-9:

                    # Create node set for boundary condition application
                    node_set = GeometrySet(node)

                    # Calculate radial displacement for compression
                    displacement = calculate_displacement_for_cylinder(
                        node.coordinates, 
                        new_radius
                    )
               
                    # Create time-dependent displacement functions for gradual compression
                    # These functions interpolate from 0 displacement at t=0 to full displacement at t=1
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

                    # Apply Dirichlet boundary condition for start nodes
                    # Constrains x,y translations and z rotation to simulate fixed marker constraint
                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                # NUMDOF 9: 3 translations + 3 rotations + 3 additional DOFs for beams
                                # ONOFF: 1=constrained, 0=free
                                # Constrain: x,y translations + z rotation for stability
                                "NUMDOF 9 ONOFF 1 1 1 0 0 1 0 0 0 "
                                "VAL 1 1 0 0 0 0 0 0 0 "
                                "FUNCT {} {} 0 0 0 0 0 0 0"  # Use displacement functions for x,y
                            ),
                            format_replacement=[displacement_x, displacement_y],
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                
                # Apply compression to nodes in the marker compression zone
                elif node.coordinates[2] < (interval[1] + interval[0]) * compressed_part:
                    # Create node set for compression boundary condition
                    node_set = GeometrySet(node)

                    # Calculate required displacement for this node's compression
                    displacement = calculate_displacement_for_cylinder(
                        node.coordinates, 
                        new_radius
                    )
                    
                    # Create time-dependent compression functions
                    displacement_x = Function(
                        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                        "VARIABLE 0 NAME a TYPE linearinterpolation "
                        "NUMPOINTS 3 TIMES 0.0 1.0 1000.0 VALUES 0.0 {} {}".format(
                            displacement[0],  # x-component of compression displacement
                            displacement[0]
                        )
                    )
                    displacement_y = Function(
                        "COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME a\n"
                        "VARIABLE 0 NAME a TYPE linearinterpolation "
                        "NUMPOINTS 3 TIMES 0.0 1.0 1000.0 VALUES 0.0 {} {}".format(
                            displacement[1],  # y-component of compression displacement
                            displacement[1]
                        )
                    )
                    mesh.add(displacement_x)
                    mesh.add(displacement_y)

                    # Apply radial compression boundary condition
                    # Only constrains x,y translations to allow axial freedom during compression
                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                "NUMDOF 9 ONOFF 1 1 0 0 0 0 0 0 0 "  # Fix only x and y translations
                                "VAL 1 1 0 0 0 0 0 0 0 "
                                "FUNCT {} {} 0 0 0 0 0 0 0"  # Use displacement functions for x,y
                            ),
                            format_replacement=[displacement_x, displacement_y],
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                    
            
    # Calculate helical geometry parameters
    tan_alpha, degrees = calculate_angle_for_intersections(n_intersections, interval, cylinder_radius)
    print(f"Helical angle: {degrees} degrees")

    # Generate the complete contour device beam structure
    print(f"Creating contour device with {number_of_wires} beam pairs...")
    create_multiple_beams_shifted_in_y(
        mesh,
        number_of_wires,
        cylinder_radius,
        interval,
        n_el,
        True,  # add_sets for boundary conditions
        mat,
        beam_object,
        tan_alpha,
        compression_factor,
        compressed_part
    )

    # Generate VTK output for visualization
    # The VTK output includes all node sets for boundary conditions visualization
    mesh.write_vtk(output_config["output_name"], base_dir)

    # Create 4C input file for finite element simulation
    # The InputFile object stores both mesh geometry and simulation parameters
    input_file = InputFile()

    # Add the beam geometry to the input file
    input_file.add(mesh)

    # Add 4C simulation parameters using config values
    # This creates a complete input file with all necessary simulation settings
    input_file.add(
        f"""
        ------------------------------------------------------------------TITLE
        Line-Based Contour Device - Configuration-driven MeshPy simulation
        -----------------------------------------------------------PROBLEM TYPE
        PROBLEMTYPE                           Structure
        RESTART                               0
        ---------------------------------------------------------------------IO
        OUTPUT_BIN                            {"yes" if output_config["binary_output"] else "no"}
        STRUCT_DISP                           {"yes" if output_config["output_displacement"] else "no"}
        FILESTEPS                             {output_config["file_steps"]}
        VERBOSITY                             {output_config["verbosity"]}
        STRUCT_STRAIN                         {"yes" if output_config["output_strain"] else "no"}
        STRUCT_STRESS                         {"yes" if output_config["output_stress"] else "no"}
        -----------------------------------------------------STRUCTURAL DYNAMIC
        LINEAR_SOLVER                         1
        INT_STRATEGY                          Standard
        DYNAMICTYPE                           {sim_config["dynamics_type"]}
        RESULTSEVERY                          {sim_config["results_every"]}
        NLNSOL                                {sim_config["solver_method"]}
        DIVERCONT                             {sim_config["divergence_control"]}
        TIMESTEP                              {time_step}
        NUMSTEP                               {num_steps}
        MAXTIME                               {sim_config["max_time"]}
        ---------------------------------------------------------------SOLVER 1
        NAME                                  Structure_Solver
        SOLVER                                {sim_config["solver_type"]}
        --------------------------------------------------IO/RUNTIME VTK OUTPUT
        OUTPUT_DATA_FORMAT                    binary
        INTERVAL_STEPS                        {output_config["interval_steps"]}
        EVERY_ITERATION                       {"yes" if output_config["every_iteration"] else "no"}
        ----------------------------------------IO/RUNTIME VTK OUTPUT/STRUCTURE
        OUTPUT_STRUCTURE                      {"yes" if output_config["output_displacement"] else "no"}
        DISPLACEMENT                          {"yes" if output_config["output_displacement"] else "no"}
        --------------------------------------------IO/RUNTIME VTK OUTPUT/BEAMS
        OUTPUT_BEAMS                          {"yes" if output_config["output_beams"] else "no"}
        DISPLACEMENT                          {"yes" if output_config["output_displacement"] else "no"}
        USE_ABSOLUTE_POSITIONS                {"yes" if output_config["absolute_positions"] else "no"}
        TRIAD_VISUALIZATIONPOINT              {"yes" if output_config["triad_visualization"] else "no"}
        STRAINS_GAUSSPOINT                    {"yes" if output_config["strain_gausspoints"] else "no"}
        ----------------------------------------------------------------BINNING STRATEGY
        BIN_SIZE_LOWER_BOUND                  {sim_config["bin_size_lower_bound"]}
        DOMAINBOUNDINGBOX                     {" ".join(map(str, sim_config["domain_bounding_box"]))}
        ----------------------------------------------------------------BEAM INTERACTION
        REPARTITIONSTRATEGY                   {beam_interaction_config["repartition_strategy"]}
        SEARCH_STRATEGY                       {beam_interaction_config["search_strategy"]}
        """
    )

    return input_file


if __name__ == "__main__":
    """
    Main execution block for line-based contour device simulation.
    
    This script performs a complete workflow:
    1. Load configuration from YAML file
    2. Create beam geometry and mesh
    3. Apply boundary conditions for compression
    4. Generate 4C input file
    5. Run finite element simulation
    
    All parameters are driven by the config.yml file, making the workflow
    fully configurable without code changes.
    """
    
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Extract configuration parameters for workflow control
    output_config = config["output"]
    parallel_config = config["parallel"]
    contour_config = config["contour"]
    
    # Ensure output directory exists for simulation results
    output_directory = contour_config["output_directory"]
    os.makedirs(output_directory, exist_ok=True)
    
    # Create the contour device input file with all geometry and parameters
    print("Generating contour device geometry and simulation parameters...")
    input_file = create_beams_wrapped_around_cylinder(config)
    
    # Write input file for 4C finite element solver
    input_file_path = os.path.join(output_directory, f"{output_config['output_name']}.dat")
    input_file.write_input_file(input_file_path)
    print(f"Input file written to: {input_file_path}")
    
    # Set simulation directory for results output
    simulation_dir = os.path.join(output_directory, "results")
    os.makedirs(simulation_dir, exist_ok=True)

    # Run 4C finite element simulation
    print(f"Starting 4C simulation with {parallel_config['num_processors']} processors...")
    return_code = run_four_c(
        input_file_path,
        simulation_dir,
        output_name=output_config["output_name"],
        n_proc=parallel_config["num_processors"]
    )
    
    if return_code == 0:
        print("Simulation completed successfully!")
        print(f"Results available in: {simulation_dir}")
    else:
        print(f"Simulation failed with return code: {return_code}")
        print("Check simulation output for error details.")