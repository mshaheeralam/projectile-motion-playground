def calculate_distance(v, angle, time):
    import math
    angle_rad = math.radians(angle)
    return v * math.cos(angle_rad) * time

def calculate_height(v, angle, time):
    import math
    angle_rad = math.radians(angle)
    return v * math.sin(angle_rad) * time - 0.5 * 9.81 * time**2

def time_of_flight(v, angle):
    import math
    angle_rad = math.radians(angle)
    return (2 * v * math.sin(angle_rad)) / 9.81

def calculate_range(v, angle):
    import math
    angle_rad = math.radians(angle)
    return (v**2 * math.sin(2 * angle_rad)) / 9.81

def linear_drag_force(velocity, drag_coefficient):
    return -drag_coefficient * velocity

def quadratic_drag_force(velocity, drag_coefficient):
    return -drag_coefficient * velocity**2

def interpolate(value, x0, y0, x1, y1):
    if x1 - x0 == 0:
        return None
    return y0 + (y1 - y0) * (value - x0) / (x1 - x0)