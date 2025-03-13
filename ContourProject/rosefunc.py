import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
a = 3  # Base amplitude (size of the rose in the xy-plane)
k = 16  # Number of petals (if k is even, number of petals = 2k; if odd, number of petals = k)
leaf_width = 50 # Controls the width of the leaves (scales the xy-plane amplitude)
z_scale = 0.1  # Scale factor for z-direction (controls how much the curve moves in z)
theta = np.linspace(0, 2 * np.pi, 1000)  # Angle from 0 to 2π

# Rose function in polar coordinates
r = a * np.cos(k * theta)

# Convert polar coordinates to Cartesian coordinates
x = leaf_width * r * np.cos(theta)  # Scale x by leaf_width
y = leaf_width * r * np.sin(theta)  # Scale y by leaf_width

# Add z-component: z increases as the curve moves away from the origin and stays at max z at the tips
z = z_scale * np.abs(r)  # z increases with distance from the origin and does not return to 0

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D rose function
ax.plot(x, y, z, color='red', linewidth=0.2)

# Set plot labels and title
ax.set_title(f'3D Rose Function with {k} Petals and Leaf Width = {leaf_width}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Calculate the maximum range for all axes
max_range = max(np.ptp(x), np.ptp(y), np.ptp(z))  # Peak-to-peak range for each axis

# Set equal scaling for all axes
ax.set_xlim([np.min(x) - 0.1 * max_range, np.max(x) + 0.1 * max_range])
ax.set_ylim([np.min(y) - 0.1 * max_range, np.max(y) + 0.1 * max_range])
ax.set_zlim([np.min(z) - 0.1 * max_range, np.max(z) + 0.1 * max_range])

# Set equal aspect ratio for 3D axes
ax.set_box_aspect([1, 1, 1])  # Ensures equal scaling for x, y, and z axes

# Save the plot as a PNG file
plt.savefig('3d_rose_function_custom_leaves.png', dpi=300, bbox_inches='tight')
