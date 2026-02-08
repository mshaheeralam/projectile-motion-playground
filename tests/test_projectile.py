import unittest
from src.physics.projectile import Projectile

class TestProjectile(unittest.TestCase):

    def setUp(self):
        self.projectile = Projectile(initial_velocity=50, launch_angle=45)

    def test_trajectory(self):
        trajectory = self.projectile.calculate_trajectory()
        self.assertIsInstance(trajectory, list)
        self.assertGreater(len(trajectory), 0)

    def test_time_of_flight(self):
        time_of_flight = self.projectile.calculate_time_of_flight()
        self.assertGreater(time_of_flight, 0)

    def test_range(self):
        range_value = self.projectile.calculate_range()
        self.assertGreater(range_value, 0)

    def test_analytical_solution(self):
        analytical_solution = self.projectile.analytical_solution()
        self.assertIsInstance(analytical_solution, tuple)

    def test_numerical_solution(self):
        numerical_solution = self.projectile.numerical_solution()
        self.assertIsInstance(numerical_solution, list)

if __name__ == '__main__':
    unittest.main()