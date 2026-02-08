class AirDrag:
    def __init__(self, drag_coefficient, area, mass):
        self.drag_coefficient = drag_coefficient
        self.area = area
        self.mass = mass

    def linear_drag_force(self, velocity):
        return self.drag_coefficient * velocity

    def quadratic_drag_force(self, velocity):
        return 0.5 * self.drag_coefficient * self.area * velocity**2

    def calculate_trajectory(self, initial_velocity, launch_angle, time_steps):
        g = 9.81  # gravitational acceleration
        trajectory = []
        vx = initial_velocity * cos(launch_angle)
        vy = initial_velocity * sin(launch_angle)

        for t in time_steps:
            # Calculate drag force
            drag_force = self.quadratic_drag_force(sqrt(vx**2 + vy**2))
            # Update velocities
            ax = -drag_force / self.mass * (vx / sqrt(vx**2 + vy**2))
            ay = -g - (drag_force / self.mass * (vy / sqrt(vx**2 + vy**2)))

            vx += ax * t
            vy += ay * t

            # Update position
            x = vx * t
            y = vy * t

            trajectory.append((x, y))

        return trajectory