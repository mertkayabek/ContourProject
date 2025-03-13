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

    def find_intersections_and_apply_coupling(mesh, beams):
        """
        Find intersections between beams and apply coupling.
        """
        node_sets = [beam["line"].get_all_nodes() for beam in beams]
        intersecting_nodes = set()

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

    mesh = Mesh()

    mat = MaterialReissner(youngs_modulus=200, radius=0.02)
    beam_object = Beam3rHerm2Line3

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
        tan_alpha = ((n_intersections - 1) * np.pi * radius) / length
        alpha_degrees = np.degrees(np.arctan(tan_alpha))
        return tan_alpha, alpha_degrees
    
    def create_multiple_beams_shifted_in_y(
        mesh,
        number_of_beams,
        cylinder_radius,
        interval,
        n_el,
        add_sets,
        material,
        beam_object,
        tan_alpha
    ):
        """
        Creates multiple beams, each shifted in y by an equal amount of 2π/number_of_beams.
        """
        def n_shape_yz(t):
            x = cylinder_radius
            y = tan_alpha * t
            z = 1.0 * t
            return npAD.array([x, y, z])
        
        def n_shape_yz2(t):
            x = cylinder_radius
            y = - tan_alpha * t
            z = 1.0 * t
            return npAD.array([x, y, z])
        
        beams = []

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

        
        mesh.wrap_around_cylinder(radius=cylinder_radius)
        #find_intersections_and_apply_coupling(mesh, beams)
        mpy.check_overlapping_elements = False
        
        # Fix start node of each beam and apply load in y direction at end of each beam
        load_val = -0.4
        #load_function = Function("SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
        #mesh.add(load_function)

        # set up displacement functions
        # x-Direction
        displacement_function_x = Function(
            "SYMBOLIC_FUNCTION_OF_SPACE_TIME {}*a*x/sqrt(x^2+y^2)\n"
            "VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 4 "
            "TIMES 0 1 2.0 1000.0 VALUES 0.0 1.0 0.0 0.0".format(load_val)
        )
        # y-Direction
        displacement_function_y = Function(
            "SYMBOLIC_FUNCTION_OF_SPACE_TIME {}*a*y/sqrt(x^2+y^2)\n"
            "VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 4 "
            "TIMES 0 1 2.0 1000.0 VALUES 0.0 1.0 0.0 0.0".format(load_val)
        )
        mesh.add(displacement_function_x)
        mesh.add(displacement_function_y)

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

            for node in all_nodes[1:middle_index]:
                x, y, z = node.coordinates
                r = np.sqrt(x**2 + y**2)
                # Calculate force components pointing towards origin (0,0)
                force_x = -x / r * load_val  # Negative sign to point towards x=0
                force_y = -y / r * load_val  # Negative sign to point towards y=0

                node_set = GeometrySet(node)

                mesh.add(
                    BoundaryCondition(
                        #beam["end"],
                        node_set,
                        (
                            "NUMDOF 9 ONOFF 1 1 0 0 0 0 0 0 0 "
                            "VAL {} {} 0 0 0 0 0 0 0 "
                            "FUNCT {} {} 0 0 0 0 0 0 0"
                        ),
                        format_replacement=[force_x, force_y, displacement_function_x, displacement_function_y],
                        bc_type=mpy.bc.neumann,
                    )
                )
            
    tan_alpha, degrees = calculate_angle_for_intersections(3, [1, 11], 1.0)

    print(f"degrees: {degrees}")

    create_multiple_beams_shifted_in_y(
        mesh,
        4,
        1.0,
        [0, 10],
        8,
        True,
        mat,
        beam_object,
        tan_alpha
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