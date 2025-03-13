import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
z_max = 4  # Amplitude (size of the petal)
k = 1  # Number of petals (k=1 for a single petal)
theta = np.linspace(0, np.pi, 100)  # Angle for one petal (0 to π)
x_max = 16  # Maximum height of the petal in x-direction
num_petals = 16  # Total number of petals to plot

# Rose function in polar coordinates
r = z_max * np.cos(k * theta)

z = r * np.cos(theta)
stretch_factor = 2  # Stretch factor for the y-axis
y = stretch_factor * r * np.sin(theta)

# Add z-direction movement (quadratic shape)
# z increases quadratically from θ=0 to θ=π/2, then decreases quadratically from θ=π/2 to θ=π
x = np.where(
    theta <= np.pi / 2,
    x_max * (2 * theta / np.pi)**2,  # Quadratic increase
    x_max * (2 * (np.pi - theta) / np.pi)**2  # Quadratic decrease
)
x= x - x_max

# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Function to rotate a petal by a given angle in the xy-plane
def rotate_petal(x, y, angle):
    # Rotation matrix for the xy-plane
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return x_rot, y_rot

# Plot the initial petal
ax.plot(x, y, z, color='red', linewidth=2, label='Initial Petal')

# Iterate to create additional petals by rotating the initial petal
for i in range(1, num_petals):
    # Calculate the rotation angle for this petal
    angle = 2 * np.pi * i / num_petals
    
    # Rotate the initial petal
    x_rot, y_rot = rotate_petal(x, y, angle)
    
    # Plot the rotated petal
    ax.plot(x_rot, y_rot, z, color='red', linewidth=2, label=f'Petal {i+1}')

# Set plot limits and labels
ax.set_xlim([-x_max, x_max])
ax.set_ylim([-x_max, x_max])
ax.set_zlim([0, z_max])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'{num_petals} 3D Rose Petals with Quadratic Z')

# Add a legend
ax.legend()
# Set the view to the YZ plane
# ax.view_init(elev=90, azim=0)
# Save the plot as a PNG file
plt.savefig('3d_rose_petals_quadratic_z_symmetric.png', dpi=300, bbox_inches='tight')