import numpy as np
import matplotlib.pyplot as plt

def draw_line_arc_line(input_angle_degrees=45):
    # Convert angle to radians
    input_angle = np.radians(input_angle_degrees)
    arc_angle = 2 * input_angle  # Total arc angle is twice the input angle
    
    # Define points
    P1 = (0, 0)
    P2 = (9, 9)  # First line end point and arc start point
    P3 = (11, 9)  # Arc end point
    P4 = (20, 0)  # Final point
    
    # Calculate radius for the arc
    chord_length = np.sqrt((P3[0] - P2[0])**2 + (P3[1] - P2[1])**2)
    radius = chord_length / (2 * np.sin(arc_angle/2))
    
    # Calculate circle center
    # Center is radius units above P2, perpendicular to the first line
    center_x = P2[0]
    center_y = P2[1] + radius
    center = (center_x, center_y)
    
    # Generate points for the arc
    # Arc starts at π and goes to π - arc_angle
    theta = np.linspace(np.pi, np.pi - arc_angle, 100)
    arc_x = center[0] + radius * np.cos(theta)
    arc_y = center[1] + radius * np.sin(theta)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw first line
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], 'b-', label='First Line')
    
    # Draw arc
    ax.plot(arc_x, arc_y, 'g-', label=f'Arc (angle: {2*input_angle_degrees}°)')
    
    # Draw second line
    ax.plot([P3[0], P4[0]], [P3[1], P4[1]], 'r-', label='Second Line')
    
    # Plot points
    ax.plot(P1[0], P1[1], 'ko', label='Start Point (0,0)')
    ax.plot(P2[0], P2[1], 'ko', label='Arc Start (9,9)')
    ax.plot(P3[0], P3[1], 'ko', label='Arc End (11,9)')
    ax.plot(P4[0], P4[1], 'ko', label='End Point (20,0)')
    ax.plot(center[0], center[1], 'go', label='Arc Center')
    
    # Add circle for reference
    circle = plt.Circle(center, radius, color='green', fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(circle)
    
    # Set plot properties
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Line-Arc-Line Transition (Arc Angle: {2*input_angle_degrees}°)')
    ax.legend()
    

    
    plt.savefig('symmetric_lines.png', dpi=300, bbox_inches='tight')

# Call the function with a specific angle
draw_line_arc_line(45)  # You can change this angle as needed

