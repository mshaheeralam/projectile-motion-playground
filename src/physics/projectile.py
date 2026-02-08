class Projectile:
    def __init__(self, initial_velocity, launch_angle):
        self.initial_velocity = initial_velocity
        self.launch_angle = launch_angle
        self.gravity = 9.81  # m/s^2

    def calculate_trajectory(self, time):
        angle_rad = self.launch_angle * (3.14159 / 180)  # Convert angle to radians
        x = self.initial_velocity * time * cos(angle_rad)
        y = (self.initial_velocity * sin(angle_rad) * time) - (0.5 * self.gravity * time ** 2)
        return x, y

    def time_of_flight(self):
        angle_rad = self.launch_angle * (3.14159 / 180)
        return (2 * self.initial_velocity * sin(angle_rad)) / self.gravity

    def range(self):
        angle_rad = self.launch_angle * (3.14159 / 180)
        return (self.initial_velocity ** 2 * sin(2 * angle_rad)) / self.gravity

    def analytical_solution(self, time):
        # Placeholder for analytical solution implementation
        pass

    def numerical_solution(self, time_step, total_time):
        # Placeholder for numerical solution implementation
        pass