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
import itertools
import glob
from datetime import datetime
import shutil
import pyvista as pv
import re
#pv.start_xvfb()  # Start virtual X server for headless rendering
import time
from datetime import timedelta


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


def create_beams_wrapped_around_cylinder(
    base_dir, 
    interval_end,
    cylinder_radius,
    compressed_part,
    beam_radius,
    positional_coupling_penalty,
    rotational_coupling_penalty,
    z_scale_factor=1.0,
    number_of_beams=12,
    youngs_modulus=83000,
    preview=False
):
    
    interval = [0, interval_end]
    n_intersections = 2
    #number_of_beams = 12
    compression_factor = 0.95
    #youngs_modulus = 83000  # Young's modulus in N/mm^2
    # austenite phase value
    # radius of marker max 0.25 mm

    n_el = 8*(n_intersections-1)*number_of_beams
    num_steps = 20
    time_step = 1/num_steps
    

    
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
        #print("\nFound close node groups:")
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
            z = R * npAD.sin(theta)  # z goes up and down
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
                # Invert the y-coordinate function to find the parameter t
                # y = tan_yz * R * (1 - cos(θ)) and θ = (t/interval[1]) * π
                # Solving for t: t = interval[1] * arccos(1 - y/(tan_yz*R)) / π
                
                # Handle potential numerical issues
                arg = 1 - y / (tan_yz * R)
                if arg < -1:
                    arg = -1
                elif arg > 1:
                    arg = 1
                    
                theta = np.arccos(arg)
                t = interval[1] * theta / np.pi
        
                # Normalize to [0, 1]
                normalized_t = t / interval[1]
                positions.append(normalized_t)
            # End with position 1
            positions.append(1.0)
            
            
            # Now add 4 extra nodes between start (0.0) and first calculated node
            
            first_interval = positions[1] - positions[0]
            for i in range(1, 5):  # Create 4 equally spaced nodes
                new_pos = positions[0] + (first_interval * i) / 5
                positions.append(new_pos)
            
            # Add 4 extra nodes between last calculated node and position 1.0
            for i in range(1, 5):  # Create 4 equally spaced nodes
                new_pos = 1 - (first_interval * i) / 5
                positions.append(new_pos)
            
            # Sort the positions to maintain proper order
            positions.sort()
            #print("Sorted node positions:", positions)

            
            return positions
        
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
            if i == 0:
                #print("\nY-coordinates of nodes on first arc:")
                beam_nodes = dir1["line"].get_all_nodes()
                beam_nodes.sort(key=lambda node: node.coordinates[1])
                # for j, node in enumerate(beam_nodes):
                # print(f"Node {j}: y = {node.coordinates[1]:.6f}")

        #mesh.display_pyvista()
        #mesh.scale(1,1,z_scale_factor)
        for node in mesh.nodes:
            node.coordinates[2] *= z_scale_factor
        mesh.wrap_around_cylinder(radius=cylinder_radius)
        find_intersections_and_apply_coupling(mesh, beams, positional_coupling_penalty, rotational_coupling_penalty) # change this
        mpy.check_overlapping_elements = False

        #print("\nBeam start and end point coordinates:")
        for i, beam in enumerate(beams):
            start_node = beam["start"].get_all_nodes()[0]  # Get the first node from the geometry set
            start_coords = start_node.coordinates
            #print(f"Beam {i + 1}: ({start_coords[0]:.3f}, {start_coords[1]:.3f}, {start_coords[2]:.3f})")
            end_node = beam["end"].get_all_nodes()[0]  # Get the first node from the geometry set
            end_coords = end_node.coordinates
            #print(f"Beam {i + 1}: ({end_coords[0]:.3f}, {end_coords[1]:.3f}, {end_coords[2]:.3f})")

        new_radius = cylinder_radius * (1.0 - compression_factor)

        # write a for loop for all nodes in mesh like below
        max_z = max(node.coordinates[2] for node in mesh.nodes if not node.is_middle_node)
        threshold = max_z * compressed_part

        for node in mesh.nodes:
            #print(f"All Node coordinates: {node.coordinates[2]}")
            if not node.is_middle_node:
                #node in beams_start: This is wrong I dont know why

                if  np.linalg.norm(node.coordinates[2] - interval[0]) < 1e-9:                #node in beams_start:
                    #print(f"Start node coordinates: {node.coordinates}")
                    # boundary condition with radial and axial=0 displacement
                    node_set = GeometrySet(node)
                    #print(f"Start Node coordinates: {node.coordinates}")
                    """
                    mesh.add(
                        BoundaryCondition(
                            node_set,
                            (
                                "NUMDOF 9 ONOFF 0 0 1 1 1 1 0 0 0 "
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
                                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 "  # Fix z also only x and y translations
                                "VAL 1 1 0 0 0 0 0 0 0 "
                                "FUNCT {} {} 0 0 0 0 0 0 0"  # Use displacement functions for x,y
                            ),
                            format_replacement=[displacement_x, displacement_y],  # Use the displacement functions
                            bc_type=mpy.bc.dirichlet,
                        )
                    )
                    
                
                elif node.coordinates[2] < threshold:
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
                                "NUMDOF 9 ONOFF 1 1 0 1 1 1 0 0 0 "  # Fix only x and y translations
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
        vtk_file_path: Path to the VTK file
        output_path: Path to save the output image
    """
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
    Copies the VTK files for specified timesteps instead of visualizing them.
    Includes both .pvtu and their associated .vtu files.
    """
    vtk_files_dir = os.path.join(results_dir, "xxx-vtk-files")
    if not os.path.exists(vtk_files_dir):
        print(f"Warning: VTK files directory not found: {vtk_files_dir}")
        return False
    
    files_copied = 0
    for timestep in keep_timesteps:
        # Format the timestep to match file pattern
        timestep_str = f"{timestep:05d}"
        
        # First find all pvtu files for this timestep (master files)
        pvtu_files = glob.glob(os.path.join(vtk_files_dir, f"structure-beams-{timestep_str}.pvtu"))
        #pvtu_files += glob.glob(os.path.join(vtk_files_dir, f"boundingbox-{timestep_str}.pvtu"))
        
        # Then find all vtu files for this timestep (piece files)
        vtu_files = glob.glob(os.path.join(vtk_files_dir, f"structure-beams-{timestep_str}-*.vtu"))
        #vtu_files += glob.glob(os.path.join(vtk_files_dir, f"boundingbox-{timestep_str}-*.vtu"))
        
        # Combine all files to copy
        all_files = pvtu_files + vtu_files
        
        if all_files:
            for src_file in all_files:
                dest_file = os.path.join(viz_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
                files_copied += 1
                #print(f"Copied: {os.path.basename(src_file)}")
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
    
    #print(f"Total files copied: {files_copied}")
    return files_copied > 0

"""
def get_dynamic_timesteps(results_dir):
   
    vtk_files_dir = os.path.join(results_dir, "xxx-vtk-files")
    if not os.path.exists(vtk_files_dir):
        print(f"Warning: VTK files directory not found: {vtk_files_dir}")
        return [0, 5, 10]  # Default fallback
    
    # Find all structure beam files
    structure_files = glob.glob(os.path.join(vtk_files_dir, "structure-beams-*.v*u"))
    
    # Extract timesteps
    timesteps = []
    for filename in structure_files:
        basename = os.path.basename(filename)
        match = re.search(r'structure-beams-(\d+)', basename)
        if match:
            timestep = int(match.group(1))
            if timestep not in timesteps:
                timesteps.append(timestep)
    
    if not timesteps:
        print("No timestep files found, using default")
        return [0, 5, 10]
    
    # Sort the timesteps
    timesteps.sort()
    
    first_timestep = 0  # Always use 0 as first
    last_timestep = timesteps[-1]
    middle_timestep = (first_timestep + last_timestep) // 2
    
    print(f"Using timesteps: first={first_timestep}, middle={middle_timestep}, last={last_timestep}")
    return [first_timestep, middle_timestep, last_timestep]
"""


def scale(self, vector):
    """Scale beam nodes of this mesh.

    Args
    ----
    vector: np.array, list
            that will be added to all nodes.
    """
    for node in self.nodes:
        node.coordinates *= vector

def parameter_sweep():
    # Define parameter ranges
    interval_end = 6.0
    cylinder_radius = 3.0
    beam_radius = 0.03
    number_of_beams = 72
    youngs_moduli = np.array([83000])  # Different material stiffness values
    beam_radii = np.array([0.02, 0.03, 0.04])  # Different beam thickness values
    compressed_parts = np.array([0.1, 0.3])  # Custom values
    positional_penalties = np.array([100])  # Custom values 
    rotational_penalties = np.array([10])  # Custom values
    z_scale_factors = np.array([0.5, 1.0])  # Custom values

    keep_timesteps = [0, 25, 50]

    # Timing variables
    sweep_start_time = time.time()

    # Path to check for existing combinations
    existing_sweeps_dir = "/home_student/kayabek/sw/Results/Sweeps"
    
    # Create base directory for results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = f"/home_student/kayabek/sw/Results/parameter_sweep_{timestamp}"
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
                    f"nbeams{number_of_beams}_"
                    f"int{interval_end:.1f}_"
                    f"rad{cylinder_radius:.2f}_"
                    f"comp{compressed_part:.2f}_"
                    f"beam{beam_radius:.2f}_"
                    f"zscale{z_scale_factor:.1f}_"
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
                        number_of_beams,
                        youngs_modulus=youngs_modulus
                    )
                    
                    input_file_path = os.path.join(params_dir, "simple_beam.dat")
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
                        max_num_steps = 120  # Max value to try
                        current_num_steps = original_num_steps + 10  # First retry with 40 steps
                        
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
                                number_of_beams,
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
                            retry_input_file_path = os.path.join(params_dir, f"simple_beam.dat")
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
                                current_num_steps += 10
                        
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

    """
    # Adapt this path to the directory you want to store the tutorial files in.
    output_directory = "/home_student/kayabek/sw/Results/arc_shape1/"
    input_file = create_beams_wrapped_around_cylinder(output_directory)
    input_file.write_input_file(os.path.join(output_directory, "simple_beam.dat"))

    simulation_dir = "/home_student/kayabek/sw/Results/arc_shape1/results"

    return_code = run_four_c(
        os.path.join(output_directory, "simple_beam.dat"),
        simulation_dir,
        output_name='xxx',
        n_proc=2 )"
    """