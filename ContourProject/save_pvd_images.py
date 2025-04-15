import pyvista as pv
import numpy as np

# Replace with your actual pvd file path
pvd_filename = "/home_student/kayabek/sw/Results/Sweeps/nbeams72_int6.0_rad3.00_comp0.40_beam0.04_zscale0.5_young60000_pos100_rot10.pvd"

# Load the pvd file using PyVista’s reader. PyVista supports pvd files by automatically reading
# the time series if available.
reader = pv.get_reader(pvd_filename)

# Get the list of time values stored in the pvd file.
# Note: Not all pvd files will expose a "time_values" attribute; consult your dataset.
time_steps = reader.time_values if hasattr(reader, 'time_values') else [0.0]
if len(time_steps) < 3:
    raise ValueError("The dataset has less than 3 time steps.")

# Define three camera-setting functions for the different views.
def set_xy_view(plotter, mesh):
    """Sets a top-down view (xy plane)."""
    # For a top view, move above the object along the z-axis.
    center = mesh.center
    # You might adjust the offset factor according to your data scale.
    camera_position = [(center[0], center[1], center[2] + 3 * mesh.length),  # position
                       center,                                             # focal point
                       (0, 1, 0)]                                          # view up
    plotter.camera_position = camera_position

def set_xz_view(plotter, mesh):
    """Sets a view along the y-axis (xz plane view)."""
    center = mesh.center
    camera_position = [(center[0], center[1] + 3 * mesh.length, center[2]),
                       center,
                       (0, 0, 1)]
    plotter.camera_position = camera_position

def set_iso_view(plotter, mesh):
    """Sets an isometric view."""
    # One common isometric perspective: offset along all three axes
    center = mesh.center
    offset = 3 * mesh.length
    camera_position = [(center[0] + offset, center[1] + offset, center[2] + offset),
                       center,
                       (0, 0, 1)]
    plotter.camera_position = camera_position

# Create a 3x3 subplot plotter. Each row corresponds to a time step, and each column to a view.
plotter = pv.Plotter(shape=(3, 3), border=False)

# For convenience, define the view functions in the order you want them to appear.
view_functions = [set_xy_view, set_xz_view, set_iso_view]

# Loop over the first three time steps (one per row).
for row, time in enumerate(time_steps[:3]):
    # Update (or load) the mesh for the current time step.
    # The reader exposes the time-specified dataset via its `update` method.
    reader.update(time)
    mesh = reader.output  # current mesh at this time step

    # Loop over the three different views for each time step.
    for col, view_func in enumerate(view_functions):
        # Switch to the designated subplot (row, col)
        plotter.subplot(row, col)
        # Clear the subplot before adding (optional if reusing a subplot).
        plotter.clear()
        # Add the mesh to the subplot.
        plotter.add_mesh(mesh, show_scalar_bar=True)

        # Set the camera position accordingly.
        view_func(plotter, mesh)
        # Reset camera clipping range and optionally call reset_camera.
        plotter.reset_camera()

# Optionally, you can link the views so zoom/panning becomes uniform, though here we used independent camera settings.
# plotter.link_views()

# Save the combined image: adjust the resolution as needed.
output_filename = "combined_3x3_output.png"
plotter.screenshot(output_filename, window_size=[1920, 1080])
plotter.close()

print(f"Saved combined image to {output_filename}")

