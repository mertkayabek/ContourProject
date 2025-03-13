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
from meshpy.mesh_creation_functions import (
    create_beam_mesh_line,
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_curve,
)

from meshpy.four_c import run_four_c
def create_two_beams_wrapped_around_cylinder(base_dir, preview=False):

    mesh = Mesh()

    mat = MaterialReissner(youngs_modulus=200, radius=0.05)
    beam_object = Beam3rHerm2Line3
    
    def half_sinusoidal(t):
        # Define the half sinusoidal shape in the y-z plane
        x = 0  # Fixed x-coordinate
        y = 4 * t
        z = npAD.sin(npAD.pi * t)
        return npAD.array([x, y, z])
    
    def half_sinusoidal_shifted(t):
        coord = half_sinusoidal(t)
        # Create a new array with the modified x-coordinate
        return npAD.array([coord[0] + 1.0, coord[1], coord[2]])
        
    # Create the first half sinusoidal shaped beam in the y-z plane
    beam1 = create_beam_mesh_curve(
        mesh, beam_object, mat,
        half_sinusoidal,
        interval=[0, 1],
        n_el=100,
        add_sets=True
    )

    # For a second beam, adjust the curve (e.g., shift z by +1 or similar)
    beam2 = create_beam_mesh_curve(
        mesh, beam_object, mat,
        half_sinusoidal_shifted,
        interval=[0, 1],
        n_el=100,
        add_sets=True
    )

    # Wrap around the z-axis (which is now the original x-axis)
    #mesh.wrap_around_cylinder(radius=5.0)  # Provide a reference radius

    mpy.check_overlapping_elements = False


    
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
    output_directory = "/home_student/kayabek/sw/meshpy/ContourProject/results_case8/"
    input_file = create_two_beams_wrapped_around_cylinder(output_directory)
    input_file.write_input_file(os.path.join(output_directory, "simple_beam.dat"))

    simulation_dir = "/home_student/kayabek/sw/meshpy/ContourProject/results_case8/results"

    return_code = run_four_c(
        os.path.join(output_directory, "simple_beam.dat"),
        simulation_dir,
        output_name='xxx',
        n_proc=2 )