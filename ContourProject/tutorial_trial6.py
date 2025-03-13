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
def simple_beam_simulation(base_dir, preview=False):
    """
    Create a honeycomb like structure with different type of connectors.

    Args
    ----
    base_dir: str
        Path where all created files will be saved.
    """

    # In the first step an empty Mesh object is created, which in general holds
    # information about the nodes, elements, materials and boundary conditions.
    # A mesh can be added to a mesh, i.e. created geometries can be combined.
    # The first geometry is created in the final mesh, as its position is
    # already final.
    mesh = Mesh()

    # We now add a straight line, with a SR beam element. If other beam
    # theories or element orders are used, simply replace the material and beam
    # objects.
    # The line is created between the two given points, with n_el elements.
    # Each mesh creation function returns certain geometry sets, where boundary
    # conditions can be applied or points can be coupled.
    mat = MaterialReissner(youngs_modulus=200, radius=0.1)
    beam_object = Beam3rHerm2Line3
    beam1 = create_beam_mesh_line(
        mesh, beam_object, mat, [0, 0, 0], [0, 1, 0], n_el=10, add_sets=True
    )
    beam2 = create_beam_mesh_line(
        mesh, beam_object, mat, [0, 1, 0], [1, 1, 0], n_el=10, add_sets=True
    )
    beam3 = create_beam_mesh_line(
        mesh, beam_object, mat, [1, 1, 0], [1, 2, 0], n_el=10, add_sets=True
    )

    # We want to fix all positions and rotations of the first node.
    # mpy is a global object that stores enums and other options for meshpy.
    mesh.add(
        BoundaryCondition(
            beam1["start"],
            (
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 "
                "VAL 0 0 0 0 0 0 0 0 0 "
                "FUNCT 0 0 0 0 0 0 0 0 0"
            ),
            bc_type=mpy.bc.dirichlet,
        )
    )

    # Define the load function
    load_val = -0.1
    load_function = Function("SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
    mesh.add(load_function)

    # Apply load to the end of the first beam in Y-direction
    mesh.add(
        BoundaryCondition(
            beam1["end"],
            (
                "NUMDOF 9 ONOFF 0 0 0 0 0 0 0 0 0 "
                "VAL 0 0 0 0 0 0 0 0 0 "
                "FUNCT 0 0 0 0 0 0 0 0 0"
            ),
            format_replacement=[load_val, load_function],
            bc_type=mpy.bc.neumann,
        )
    )

    
    mesh.add(
        BoundaryCondition(
            beam2["end"],
            (
                "NUMDOF 9 ONOFF 0 0 0 0 0 0 0 0 0 "
                "VAL 0 0 0 0 0 0 0 0 0 "
                "FUNCT 0 0 0 0 0 0 0 0 0"
            ),
            format_replacement=[load_val, load_function],
            bc_type=mpy.bc.neumann,
        )
    )

    # Apply load to the end of the third beam in Y-direction
    mesh.add(
        BoundaryCondition(
            beam3["end"],
            (
                "NUMDOF 9 ONOFF 1 0 0 0 0 0 0 0 0 "
                "VAL {} 0 0 0 0 0 0 0 0 "
                "FUNCT {} 0 0 0 0 0 0 0 0"
            ),
            format_replacement=[load_val, load_function],
            bc_type=mpy.bc.dirichlet,
        )
    )

    end_node1   = beam1["end"].get_all_nodes()[0]
    start_node2 = beam2["start"].get_all_nodes()[0]
    end_node2   = beam2["end"].get_all_nodes()[0]
    start_node3 = beam3["start"].get_all_nodes()[0]


    mesh.couple_nodes(
        nodes=[end_node1, start_node2],
        coupling_type=mpy.bc.point_coupling_penalty,
        coupling_dof_type="POSITIONAL_PENALTY_PARAMETER 1000 ROTATIONAL_PENALTY_PARAMETER 1000"
    )

    mesh.couple_nodes(
        nodes=[end_node2, start_node3],
        coupling_type=mpy.bc.point_coupling_penalty,
        coupling_dof_type="POSITIONAL_PENALTY_PARAMETER 1000 ROTATIONAL_PENALTY_PARAMETER 1000"
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
    output_directory = "/home_student/kayabek/sw/meshpy/ContourProject/results_case6/"
    input_file = simple_beam_simulation(output_directory)
    input_file.write_input_file(os.path.join(output_directory, "simple_beam.dat"))

    simulation_dir = "/home_student/kayabek/sw/meshpy/ContourProject/results_case6/results"

    return_code = run_four_c(
        os.path.join(output_directory, "simple_beam.dat"),
        simulation_dir,
        output_name='xxx',
        n_proc=2 )