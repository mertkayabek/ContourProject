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
Rose Petal Medical Contour Device Generator

This script generates a medical contour device with parametric rose petal geometry.
The device consists of multiple beam elements arranged in a rose petal pattern,
designed for medical applications.

Author: Mert Kayabek
"""

import numpy as np
import autograd.numpy as npAD
import os
import yaml

# MeshPy core modules
from meshpy.core.conf import mpy
from meshpy.core.geometry_set import GeometrySet
from meshpy.core.mesh import Mesh

# MeshPy beam and material modules
from meshpy.four_c.boundary_condition import BoundaryCondition
from meshpy.four_c.element_beam import Beam3rHerm2Line3
from meshpy.four_c.function import Function
from meshpy.four_c.input_file import InputFile
from meshpy.four_c.material import MaterialReissner
from meshpy.four_c.run_four_c import run_four_c

# MeshPy utility modules
from meshpy.mesh_creation_functions.beam_curve import create_beam_mesh_curve

def create_rose_petals(config):
    """
    Create a rose petal medical contour device with parametric geometry.
    
    This function generates a medical device consisting of multiple beam elements
    arranged in a rose petal pattern. Each petal follows a parametric curve that creates a smooth shape.
    
    Args:
        config (dict): Configuration dictionary containing all parameters
        
    Returns:
        InputFile: Complete 4C input file ready for simulation
    """
    # Initialize mesh and extract parameters from config
    mesh = Mesh()
    
    # Extract configuration parameters
    geom_config = config["geometry"]
    material_config = config["material"]
    mesh_config = config["mesh"]
    bc_config = config["boundary_conditions"]
    output_config = config["output"]
    
    # Material definition
    material = MaterialReissner(
        youngs_modulus=material_config["youngs_modulus"], 
        radius=material_config["radius"]
    )
    beam_class = Beam3rHerm2Line3

    # Geometric parameters
    z_max = geom_config["z_max"]                    # Maximum height of the petal
    x_max = geom_config["x_max"]                    # Maximum width of the petal
    stretch_factor = geom_config["stretch_factor"]  # Y-direction stretch factor
    num_petals = geom_config["num_petals"]          # Number of petals

    # Mesh parameters
    n_elements = mesh_config["elements_per_petal"]  # Number of elements per petal

    def create_petals():
        """
        Generate all petals with their parametric shapes.
        
        Returns:
            list: List of beam dictionaries, each containing petal geometry
        """
        beams = []  # Store generated beams for each petal
        
        # Create petals at different rotational angles
        for i in range(num_petals):
            # Calculate rotation angle for this petal
            angle = 2 * npAD.pi * i / num_petals

            def shape(t, angle=angle):
                """
                Define the parametric shape function for a single petal.
                
                This function creates a smooth, organic petal shape using:
                - Parametric angle ranging from 0 to π
                - Cosine-based radial shaping
                - Catheter integration with quick ascent/descent
                - Quadratic x-coordinate shaping for natural curvature
                
                Args:
                    t (float): Parameter from 0 to 1
                    angle (float): Rotation angle for this petal
                    
                Returns:
                    ndarray: 3D coordinates [x, y, z] of the petal curve
                """
                # Map parameter t (0->1) to theta (0->π)
                theta = t * npAD.pi
                
                # Radial component based on cosine for smooth variation
                r = z_max * npAD.cos(theta)
                
                # Z-coordinate system with catheter integration
                # Creates quick ascent/descent near catheter using sine³ for steeper slopes
                z = ((1- npAD.sin(theta)**3) +  # Base catheter depth
                     z_max * npAD.sin(theta)**2)                    # Peak elevation
                
                # Y-coordinate stretched by factor to modify petal shape
                y_original = stretch_factor * r * npAD.sin(theta)
                
                # X-coordinate with quadratic shaping for natural curvature
                x_original = npAD.where(
                    theta <= npAD.pi/2,
                    x_max * (2 * theta / npAD.pi)**2,              # Quadratic increase
                    x_max * (2 * (npAD.pi - theta)/npAD.pi)**2     # Quadratic decrease
                )
                x_original -= x_max  # Shift to start at -x_max
                
                # Apply rotation to create radial petal arrangement
                x_rot = x_original * npAD.cos(angle) - y_original * npAD.sin(angle)
                y_rot = x_original * npAD.sin(angle) + y_original * npAD.cos(angle)
                
                return npAD.array([x_rot, y_rot, z])

            # Create beam with parametric shape
            beam = create_beam_mesh_curve(
                mesh,
                beam_class,
                material,
                shape,
                interval=[0, 1],
                n_el=n_elements,
                add_sets=True
            )
            beams.append(beam)
            
        return beams

    # Generate all petals
    beams = create_petals()

    # Apply boundary conditions if enabled
    if bc_config["fix_base_points"]:
        # Fix base points (start and end of each petal) to prevent rigid body motion
        for beam in beams:
            for end in [beam["start"], beam["end"]]:
                mesh.add(
                    BoundaryCondition(
                        end,
                        "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                        bc_type=mpy.bc.dirichlet
                    )
                )

    # Generate VTK output for visualization if enabled
    if output_config["vtk_output"]:
        base_dir = output_config["base_directory"]
        mesh.write_vtk("rose_petal_device", base_dir)

    # Create 4C input file
    input_file = InputFile()
    input_file.add(mesh)
    
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
        TIMESTEP                              1
        NUMSTEP                               1
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
        """,
        option_overwrite=True,
    )

    return input_file


def load_config(config_path):
    """
    Load configuration parameters from YAML file.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


if __name__ == "__main__":
    """
    Main execution section for the Rose Petal Medical Contour Device generator.
    
    This script:
    1. Loads configuration from YAML file
    2. Generates the rose petal device geometry
    3. Creates simulation input files
    4. Runs the 4C simulation
    """
    
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    config = load_config(config_path)
    
    # Extract output directories from config
    output_config = config["output"]
    parallel_config = config["parallel"]
    
    output_directory = output_config["base_directory"]
    simulation_dir = output_config["simulation_directory"]
    output_name = output_config["output_name"]
    
    # Ensure output directories exist
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(simulation_dir, exist_ok=True)
    
    print("=" * 60)
    print("Rose Petal Medical Contour Device Generator")
    print("Author: Mert Kayabek")
    print("=" * 60)
    print(f"Configuration loaded from: {config_path}")
    print(f"Number of petals: {config['geometry']['num_petals']}")
    print(f"Output directory: {output_directory}")
    print(f"Simulation directory: {simulation_dir}")
    print("=" * 60)
    
    # Generate the rose petal device
    print("Generating rose petal device geometry...")
    input_file = create_rose_petals(config)
    
    # Write input file
    input_file_path = os.path.join(output_directory, f"{output_name}.dat")
    input_file.write_input_file(input_file_path)
    print(f"Input file written to: {input_file_path}")
    
    # Run 4C simulation
    print("Starting 4C simulation...")
    return_code = run_four_c(
        input_file_path,
        simulation_dir,
        output_name=output_name,
        n_proc=parallel_config["num_processors"]
    )
    
    if return_code == 0:
        print("Simulation completed successfully!")
    else:
        print(f"Simulation failed with return code: {return_code}")
    
    print("=" * 60)