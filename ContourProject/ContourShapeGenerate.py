import numpy as np
import autograd.numpy as npAD
import matplotlib.pyplot as plt

# ======================
# 1. PARAMETER SETUP
# ======================
z_max = 8            # Maximum height of petals
catheter_depth = 2   # Depth at catheter connection
num_petals = 16      # Number of petals around Z-axis
x_max = 16           # Horizontal spread control
stretch_factor = 2   # Y-axis stretching factor

# Angular parametrization (0 to π)
theta = np.linspace(0, np.pi, 1000)


# Define the original function f(theta)
def f(theta):
    return -catheter_depth * (1 - npAD.sin(theta)**3) + z_max * npAD.sin(theta)**2

# Define the two transition boundaries:
theta_low = 7 * npAD.pi / 16
theta_high = 9 * npAD.pi / 16

# Precompute the endpoint values
f_low = f(theta_low)
f_high = f(theta_high)

# Desired mid-point value at theta = pi/2
theta_mid = npAD.pi/2
f_mid = -catheter_depth

# Reparameterize: u = (theta - theta_low)/(theta_high-theta_low), u in [0,1]
def h(u, a, b, c, d):
    return a + b*u + c*u**2 + d*u**3

# We want to determine coefficients a, b, c, d such that:
# (1) h(0) = a = f_low
# (2) h(1) = a + b + c + d = f_high
# (3) h(0.5) = a + 0.5b + 0.25c + 0.125d = f_mid

# We have three conditions and four unknowns. We can choose one free parameter.
# For example, let’s set the derivative at u=0.5 to a desired value S.
# Suppose we want S to be some nonzero value, e.g., S = (f_high - f_low) (a rough scale).
S = (f_high - f_low)  # you can adjust S as desired

# The derivative of h(u) is: h'(u)= b + 2c*u + 3d*u**2.
# Condition (4): h'(0.5)= b + c + 0.75d = S

# Now, we have:
# (1) a = f_low
# (2) b + c + d = f_high - f_low
# (3) 0.5b + 0.25c + 0.125d = f_mid - f_low
# (4) b + c + 0.75d = S

# Let a = f_low.
# From (2): b + c = (f_high - f_low) - d.
# From (3): Multiply by 8: 4b + 2c + d = 8(f_mid - f_low)  -> 4b+2c = 8(f_mid - f_low)-d.
# Divide by 2: 2b + c = 4(f_mid - f_low) - d.
# Now, subtract (2): (2b + c) - (b + c) = [4(f_mid - f_low) - d] - [(f_high - f_low) - d],
# so: b = 4(f_mid - f_low) - (f_high - f_low) = 4f_mid - 4f_low - f_high + f_low = 4f_mid - f_high - 3f_low.
# Then, from (2): c = (f_high - f_low) - d - b.
# Finally, use (4) to determine d:
# (4) gives: b + c + 0.75d = S  => (b + [(f_high - f_low) - d - b]) + 0.75d = S,
# which simplifies to: (f_high - f_low) - d + 0.75d = S  ->  (f_high - f_low) - 0.25d = S,
# so: d = 4*((f_high - f_low) - S).

a_coeff = f_low
b_coeff = 4*f_mid - f_high - 3*f_low
d_coeff = 4*((f_high - f_low) - S)
c_coeff = (f_high - f_low) - d_coeff - b_coeff

# Now, for theta values, define a piecewise function for z:
z = npAD.empty_like(theta)
for i, th in enumerate(theta):
    if th <= theta_low:
        # Use the original function f
        z[i] = f(th)
    elif th >= theta_high:
        # Use the original function f
        z[i] = f(th)
    else:
        # Use the cubic interpolation in the interval [theta_low, theta_high]
        u = (th - theta_low) / (theta_high - theta_low)
        z[i] = h(u, a_coeff, b_coeff, c_coeff, d_coeff)
                     
# ======================
# 3. PLANAR COORDINATES
# ======================
# Radial component
r = z_max * np.cos(theta)

# X-coordinate with quadratic shaping
x_original = np.where(
    theta <= np.pi/2,
    x_max * (2*theta/np.pi)**2,
    x_max * (2*(np.pi - theta)/np.pi)**2
) - x_max  # Center around origin

# Y-coordinate with stretching
y_original = stretch_factor * r * np.sin(theta)

# ======================
# 4. 3D VISUALIZATION
# ======================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot each petal with radial rotation
for i in range(num_petals):
    angle = 2 * np.pi * i / num_petals
    
    # Rotate coordinates around Z-axis
    x_rot = x_original * np.cos(angle) - y_original * np.sin(angle)
    y_rot = x_original * np.sin(angle) + y_original * np.cos(angle)
    
    # Create color gradient effect
    ax.plot(x_rot, y_rot, z, 
            color=plt.cm.plasma(i/num_petals),
            linewidth=2.5,
            alpha=0.8)

# Configure plot aesthetics
ax.set_xlim([-x_max*1.1, x_max*1.1])
ax.set_ylim([-x_max*1.1, x_max*1.1])
ax.set_zlim([-catheter_depth-1, z_max+1])
ax.view_init(elev=25, azim=-45)
ax.grid(False)

# Labels and title
ax.set_xlabel('X Axis', fontsize=12, labelpad=15)
ax.set_ylabel('Y Axis', fontsize=12, labelpad=15)
ax.set_zlabel('Z Axis', fontsize=12, labelpad=15)
ax.set_title('3D Vascular Coil Shape',
            fontsize=14, pad=20)

plt.tight_layout()
plt.show()

plt.savefig('ContourShape1.png', dpi=300, bbox_inches='tight')