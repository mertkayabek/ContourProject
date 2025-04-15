import pyvista as pv
from PIL import Image  # Add this import
import os
import glob

# Define the folder path where the .pvd files are located
folder_path = "/root/thesis/backup/Results/Sweeps"

# 🔍 Get all .pvd files
pvd_files = glob.glob(os.path.join(folder_path, "*.pvd"))

# Helper to render and save individual view images
def render_view(mesh, view='xy', idx=0, file_tag=""):
    plotter = pv.Plotter(off_screen=True)
    plotter.enable_parallel_projection()
    plotter.add_mesh(mesh, show_scalar_bar=True)

    center = mesh.center
    offset = 1.5 * mesh.length

    if view == 'xy':
        camera_position = [(center[0], center[1], center[2] + offset), center, (0, 1, 0)]
    elif view == 'yz':
        camera_position = [(center[0] + offset, center[1], center[2]), center, (0, 0, 1)]
    elif view == 'iso':
        camera_position = [(center[0] + offset, center[1] + offset, center[2] + offset), center, (0, 0, 1)]
    else:
        raise ValueError("Unknown view type")

    plotter.camera_position = camera_position
    plotter.camera.zoom(0.7)
    plotter.reset_camera()

    filename = f"temp_{file_tag}_{view}_{idx}.png"
    plotter.screenshot(filename, window_size=[800, 800])
    plotter.close()
    return filename

# Loop through all .pvd files
for pvd_path in pvd_files:
    base_name = os.path.splitext(os.path.basename(pvd_path))[0]
    print(f"🔄 Processing: {base_name}")

    reader = pv.get_reader(pvd_path)
    time_steps = reader.time_values if hasattr(reader, 'time_values') else [0.0]
    selected_times = time_steps[:3]

    temp_files = []

    # Render 3 views for 3 timesteps → 9 images
    rows = []
    for i, time in enumerate(selected_times):
        reader.set_active_time_value(time)
        mesh = reader.read()

        xy_img = render_view(mesh, view='xy', idx=i, file_tag=base_name)
        yz_img = render_view(mesh, view='yz', idx=i, file_tag=base_name)
        iso_img = render_view(mesh, view='iso', idx=i, file_tag=base_name)

        temp_files.extend([xy_img, yz_img, iso_img])
        rows.append((xy_img, yz_img, iso_img))

    # Combine row images horizontally
    row_images = []
    for img_paths in rows:
        imgs = [Image.open(p) for p in img_paths]
        total_width = sum(img.width for img in imgs)
        max_height = max(img.height for img in imgs)

        row = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in imgs:
            row.paste(img, (x_offset, 0))
            x_offset += img.width
        row_images.append(row)

    # Combine all rows vertically
    final_width = row_images[0].width
    final_height = sum(r.height for r in row_images)
    final_image = Image.new("RGB", (final_width, final_height))

    y_offset = 0
    for row in row_images:
        final_image.paste(row, (0, y_offset))
        y_offset += row.height

    # Save final output image
    output_path = os.path.join(folder_path, f"{base_name}.png")
    final_image.save(output_path)
    print(f"✅ Saved image: {output_path}")

    # Cleanup temp images
    for tmp in temp_files:
        os.remove(tmp)
    print("🧹 Temp files cleaned up\n")