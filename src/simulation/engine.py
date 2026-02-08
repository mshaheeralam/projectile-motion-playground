class SimulationEngine:
    def __init__(self, projectile, air_drag=None):
        self.projectile = projectile
        self.air_drag = air_drag

    def simulate(self, time_step, total_time):
        trajectory = []
        time = 0.0

        while time <= total_time:
            position = self.projectile.calculate_position(time)
            trajectory.append(position)

            if self.air_drag:
                self.air_drag.apply_drag(self.projectile)

            time += time_step

        return trajectory

    def reset(self, new_projectile):
        self.projectile = new_projectile
        if self.air_drag:
            self.air_drag.reset()