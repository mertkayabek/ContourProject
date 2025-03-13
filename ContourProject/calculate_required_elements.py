import numpy as np

def calculate_required_elements(cylinder_radius, tan_yz, number_of_beams, interval):
    """
    Calculate intersection points of beams after wrapping around cylinder.
    
    Args:
        cylinder_radius: float
            Radius of the cylinder
        tan_yz: float
            Tangent of angle between y and z axes (y/z ratio)
        number_of_beams: int
            Number of beam pairs
        interval: list
            [start, end] interval of the beam
            
    Returns:
        tuple: (n_el, z_points, intersection_points)
            n_el: int - Minimum number of elements needed
            z_points: list - Sorted z-coordinates of intersection points
            intersection_points: list - (theta, z) coordinates of intersections
    """
    intersection_points = []
    
    for i in range(number_of_beams):
        shift_i = (2.0 * np.pi / number_of_beams) * i
        
        for j in range(number_of_beams):
            shift_j = (2.0 * np.pi / number_of_beams) * j
            
            t_intersect = ((shift_j - shift_i) * cylinder_radius) / (2 * tan_yz)
            
            interval_length = interval[1] - interval[0]
            shifts_needed = np.ceil((interval[0] - t_intersect) / interval_length)
            t_intersect += shifts_needed * interval_length
            # t_intersect = (shift_j - shift_i) * cylinder_radius / (2 * tan_yz)
            z_intersect = t_intersect
            theta_intersect = shift_i + np.arctan(tan_yz * t_intersect / cylinder_radius)
            
             # Debug print to see what's being calculated
            print(f"Checking intersection:")
            print(f"  Beam 1 shift: {shift_i:.3f}")
            print(f"  Beam 2 shift: {shift_j:.3f}")
            print(f"  t_intersect: {t_intersect:.3f}")
            print(f"  z_intersect: {z_intersect:.3f}")
            print(f"  theta_intersect: {theta_intersect:.3f}")
            print(f"  Interval check: {interval[0]} <= {z_intersect} <= {interval[1]}")

            if interval[0] <= z_intersect <= interval[1]:
                intersection_points.append((theta_intersect, z_intersect))
                print(f"  Added intersection point!")

    z_points = [p[1] for p in intersection_points]
    total_length = interval[1] - interval[0]
    
    # Find smallest distance between intersections
    z_points.sort()
    min_distance = total_length
    for i in range(len(z_points)-1):
        distance = z_points[i+1] - z_points[i]
        min_distance = min(min_distance, distance)
    
    n_el = int(np.ceil(total_length / min_distance))
    
    return n_el, z_points, intersection_points

def main():
    # Input parameters
    cylinder_radius = 1.0  # Radius of cylinder
    tan_yz = 4.0          # Tangent of angle between y and z axes
    number_of_beams = 2   # Number of beam pairs
    interval = [1, 11]    # Beam length interval [start, end]
    
    # Calculate required elements and intersection points
    n_el, z_points, intersection_points = calculate_required_elements(
        cylinder_radius, tan_yz, number_of_beams, interval
    )

    # Print debug information
    print("\nInput parameters:")
    print(f"Cylinder radius: {cylinder_radius}")
    print(f"Tangent (y/z): {tan_yz}")
    print(f"Number of beams: {number_of_beams}")
    print(f"Interval: {interval}")
    
    # Print results
    print(f"\nCalculation Results:")
    print(f"Required number of elements (n_el): {n_el}")
    print(f"\nZ-coordinates of intersection points:")
    for i, z in enumerate(z_points):
        print(f"Point {i+1}: z = {z:.3f}")
    
    print(f"\nAll intersection points (theta, z):")
    for i, point in enumerate(intersection_points):
        print(f"Point {i+1}: theta = {point[0]:.3f}, z = {point[1]:.3f}")

if __name__ == "__main__":
    main()