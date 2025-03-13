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

import numpy as np
import autograd.numpy as npAD
import os
from meshpy import (
    mpy, Mesh, MaterialReissner, Beam3rHerm2Line3, BoundaryCondition,
    Function, GeometrySet, InputFile
)
from meshpy.mesh_creation_functions import create_beam_mesh_curve
from meshpy.four_c import run_four_c

def create_rose_petals(base_dir, preview=False):
    mesh = Mesh()
    material = MaterialReissner(youngs_modulus=200, radius=0.5)
    beam_class = Beam3rHerm2Line3

    # Parameters from original code
    z_max = 8 # Maximum height of the petal
    x_max = 8 # Maximum width of the petal
    stretch_factor = 1 # Factor to stretch the petal in the y-direction
    catheter_depth = 1  # Depth of microcatheter below z=0
    num_petals = 2 # Number of petals to create
    mpy.check_overlapping_elements=False
    # Check out arctan for the shape it has 2 parameters to set

    """
    def find_intersections_and_apply_coupling(mesh, beams):
        
        #Find intersections between beams and apply coupling.
        
        node_sets = [beam["line"].get_all_nodes() for beam in beams]
        intersecting_nodes = set()

        # Find intersecting nodes
        for i in range(len(node_sets)):
            for j in range(i + 1, len(node_sets)):
                for node_i in node_sets[i]:
                    for node_j in node_sets[j]:
                        if np.allclose(node_i.coordinates, node_j.coordinates):
                            intersecting_nodes.add((node_i, node_j))

        # Apply coupling to intersecting nodes
        for node_i, node_j in intersecting_nodes:
            mesh.couple_nodes(
                nodes=[node_i, node_j],
                coupling_type=mpy.bc.point_coupling_penalty,
                coupling_dof_type="POSITIONAL_PENALTY_PARAMETER 200 ROTATIONAL_PENALTY_PARAMETER 0"
            )
    """
    def create_petals():
        beams = [] # List to store the generated beams for each petal
        
        # Loop through the number of petals, placing each at a different rotational angle
        for i in range(num_petals):
            angle = 2 * npAD.pi * i / num_petals

            def shape(t, angle=angle):
                # Parametric angle ranging from 0 to pi
                theta = t * npAD.pi  # t: 0->1 ~ theta: 0->pi
                # Radial and z-coordinate components (derived from cosine shaping)
                r = z_max * npAD.cos(theta)
                # NEW Z-COORDINATE SYSTEM: Catheter integration
                # ---------------------------------------------
                # Quick ascent/descent near catheter (theta near 0/π)
                # Smooth transition using sine^3 for steeper slopes
                z = (-catheter_depth * (1 - npAD.sin(theta)**3) +  # Base catheter depth
                     z_max * npAD.sin(theta)**2)                    # Peak elevation
                # y-coordinate, stretched by a factor to modify petal's shape
                y_original = stretch_factor * r * npAD.sin(theta)
                
                # Compute x_original with quadratic shaping
                x_original = npAD.where(
                    theta <= npAD.pi/2,
                    x_max * (2 * theta / npAD.pi)**2, # Quadratic increase
                    x_max * (2 * (npAD.pi - theta)/npAD.pi)**2 # Quadratic decrease
                )
                x_original -= x_max  # Shift to start at -x_max
                
                # Rotate the x and y coordinates by the specified angle for this petal
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
                n_el=50,
                add_sets=True
            )
            beams.append(beam)
        return beams

    # Generate all petals
    beams = create_petals()
    print("Number of beams generated:", len(beams))
    #find_intersections_and_apply_coupling(mesh, beams)

    # Couple nodes at the tip (0,0,0)
    """
    tip_nodes = []
    for beam in beams:
        for node in beam["line"].get_all_nodes():
            if np.allclose(node.coordinates, [0, 0, 0], atol=1e-6):
                tip_nodes.append(node)
    
    if tip_nodes:
        mesh.couple_nodes(
            nodes=tip_nodes,
            coupling_type=mpy.bc.point_coupling_penalty,
            coupling_dof_type="POSITIONAL_PENALTY_PARAMETER 200 ROTATIONAL_PENALTY_PARAMETER 0"
        )
    """
    mid_nodes = []  # List to store the middle node from each beam
    for beam in beams:
        nodes = beam["line"].get_all_nodes()  # Get all nodes in the beam
        mid_index = len(nodes) // 2           # Compute the middle index (integer division)
        mid_nodes.append(nodes[mid_index])    # Append the middle node
    # Fix base points (start and end of each petal)

    load_val = -0.4
    load_function = Function("SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
    mesh.add(load_function)
    for beam in beams:
        nodes = beam["line"].get_all_nodes()
        mid_index = len(nodes) // 2
        middle_node = nodes[mid_index]
        neighbors = []
        if mid_index > 0:
            neighbors.append(nodes[mid_index - 1])
        neighbors.append(nodes[mid_index])
        if mid_index < len(nodes) - 1:
            neighbors.append(nodes[mid_index + 1])
        middle_node_set = GeometrySet(neighbors)
        middle_node_set.name = "middle"
        mesh.add(
            BoundaryCondition(
                beam["start"],
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 {} 0 0 0 0 0 0 FUNCT 0 0 {} 0 0 0 0 0 0",
                format_replacement=[load_val, load_function],
                bc_type=mpy.bc.dirichlet
            )
        )
        mesh.add(
            BoundaryCondition(
                middle_node_set,
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
                #format_replacement=[load_val, load_function],
                bc_type=mpy.bc.dirichlet
            )
        )

    # Apply load at the tip
    
    
    #mesh.add(load_function)
    #load_function_index = load_function.get_global_index()  # Ensure the function has an index
    """
    #if mid_nodes:
        mesh.add(
            BoundaryCondition(
                GeometrySet(mid_nodes),
                f"NUMDOF 9 ONOFF 1 1 0 0 0 0 0 0 0 VAL {} {} 0 0 0 0 0 0 0 FUNCT {} 0 0 0 0 0 0 0 0",
                format_replacement=[load_val, load_function],
                bc_type=mpy.bc.neumann
            )
        )
    """
    """
    for beam in beams:
        mesh.add(
                    BoundaryCondition(
                        beam["start"],
                        (
                            "NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 "
                            "VAL 0 {} 0 0 0 0 0 0 0 "
                            "FUNCT 0 {} 0 0 0 0 0 0 0"
                        ),
                        format_replacement=[load_val, load_function],
                        bc_type=mpy.bc.dirichlet,
                    )
                )
    """
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
    output_directory = "/home_student/kayabek/sw/meshpy/ContourProject/results_case11/"
    input_file = create_rose_petals(output_directory)
    input_file.write_input_file(os.path.join(output_directory, "simple_beam.dat"))

    simulation_dir = "/home_student/kayabek/sw/meshpy/ContourProject/results_case11/results"

    return_code = run_four_c(
        os.path.join(output_directory, "simple_beam.dat"),
        simulation_dir,
        output_name='xxx',
        n_proc=2 )