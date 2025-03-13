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
from meshpy.utility import get_single_node
from meshpy.mesh_creation_functions import (
    create_beam_mesh_line,
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_curve,
)


def simple_beam_simulation(base_dir):
    """

    Simulate a simple 2D beam fixed at the left side and loaded at the right edge.
    
"""
    # Create the mesh object
    mesh = Mesh()

    # Define beam material and element type
    mat = MaterialReissner(youngs_modulus=1e7, radius=0.02)
    beam_object = Beam3rHerm2Line3

    # Create a single horizontal beam (1m long, divided into 10 elements)
    beam_set = create_beam_mesh_line(
        mesh, beam_object, mat, [0, 0, 0], [1.0, 0, 0], n_el=10
    )

    # Fix the left end of the beam (all DOFs)
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            (
                "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 "
                "FUNCT 0 0 0 0 0 0 0 0 0"
            ),
            bc_type=mpy.bc.dirichlet,
        )
    )

    # Allow free vertical displacement at the right end (fix horizontal displacement and rotation)
    mesh.add(
        BoundaryCondition(
            beam_set["end"],
            (
                "NUMDOF 9 ONOFF 0 0 0 0 0 0 0 0 0 VAL 0 0 0 0 0 0 0 0 0 "
                "FUNCT 0 0 0 0 0 0 0 0 0"
            ),
            bc_type=mpy.bc.dirichlet,
        )
    )

    # Apply a vertical load at the right end (-1000 N in y-direction)
    line_load_val = 0.00000001
    load_function = Function("COMPONENT 0 SYMBOLIC_FUNCTION_OF_SPACE_TIME t")
    mesh.add(load_function)
    mesh.add(
        BoundaryCondition(
            beam_set["end"],
            (
                "NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 -1000 0 0 0 0 0 0 0 "
                "FUNCT 0 0 0 0 0 0 0 0 0"
            ),
            format_replacement=[line_load_val, load_function],
            bc_type=mpy.bc.neumann,
        )
    )

    # Write mesh to VTK for visualization
    mesh.write_vtk("simple_beam", base_dir)

    # Create input file for the solver
    input_file = InputFile()
    input_file.add(mesh)
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
        """
    )

    return input_file


if __name__ == "__main__":
    """Execution part of script."""

    # Adapt this path to the directory you want to store the tutorial files in.
    output_directory = "/home_student/kayabek/sw/meshpy/ContourProject/results_simple_beam/"
    input_file = simple_beam_simulation(output_directory)
    input_file.write_input_file(os.path.join(output_directory, "simple_beam.dat"))
