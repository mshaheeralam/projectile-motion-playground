import unittest
from src.physics.air_drag import AirDrag

class TestAirDrag(unittest.TestCase):

    def setUp(self):
        self.air_drag = AirDrag()

    def test_trajectory_with_linear_drag(self):
        initial_velocity = 50  # m/s
        angle = 45  # degrees
        drag_coefficient = 0.1  # linear drag coefficient
        expected_trajectory = self.air_drag.calculate_trajectory(initial_velocity, angle, drag_coefficient)
        self.assertIsNotNone(expected_trajectory)

    def test_trajectory_with_quadratic_drag(self):
        initial_velocity = 50  # m/s
        angle = 45  # degrees
        drag_coefficient = 0.05  # quadratic drag coefficient
        expected_trajectory = self.air_drag.calculate_trajectory(initial_velocity, angle, drag_coefficient, drag_type='quadratic')
        self.assertIsNotNone(expected_trajectory)

    def test_trajectory_with_no_drag(self):
        initial_velocity = 50  # m/s
        angle = 45  # degrees
        expected_trajectory = self.air_drag.calculate_trajectory(initial_velocity, angle)
        self.assertIsNotNone(expected_trajectory)

if __name__ == '__main__':
    unittest.main()