"""
Tests for projectile motion physics (without air resistance).
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.projectile import (
    ProjectileMotion, TrajectoryPoint,
    calculate_range, calculate_max_height, calculate_flight_time
)
from physics.constants import G


class TestProjectileMotion:
    """Test suite for ProjectileMotion class."""
    
    def test_init_velocity_components(self):
        """Test that initial velocity is correctly decomposed."""
        # 45 degrees should give equal vx and vy
        proj = ProjectileMotion(v0=10.0, angle=45.0)
        expected = 10.0 * np.cos(np.radians(45))
        
        assert np.isclose(proj.vx0, expected, rtol=1e-10)
        assert np.isclose(proj.vy0, expected, rtol=1e-10)
    
    def test_init_horizontal_launch(self):
        """Test horizontal launch (0 degrees)."""
        proj = ProjectileMotion(v0=20.0, angle=0.0)
        
        assert np.isclose(proj.vx0, 20.0, rtol=1e-10)
        assert np.isclose(proj.vy0, 0.0, atol=1e-10)
    
    def test_init_vertical_launch(self):
        """Test vertical launch (90 degrees)."""
        proj = ProjectileMotion(v0=20.0, angle=90.0)
        
        assert np.isclose(proj.vx0, 0.0, atol=1e-10)
        assert np.isclose(proj.vy0, 20.0, rtol=1e-10)


class TestAnalyticalSolutions:
    """Test analytical (closed-form) solutions."""
    
    def test_position_at_t0(self):
        """Position at t=0 should be initial position."""
        proj = ProjectileMotion(v0=50.0, angle=45.0, x0=0.0, y0=0.0)
        x, y = proj.position_analytical(0.0)
        
        assert x == 0.0
        assert y == 0.0
    
    def test_velocity_at_t0(self):
        """Velocity at t=0 should be initial velocity."""
        proj = ProjectileMotion(v0=50.0, angle=45.0)
        vx, vy = proj.velocity_analytical(0.0)
        
        assert np.isclose(vx, proj.vx0)
        assert np.isclose(vy, proj.vy0)
    
    def test_max_height_formula(self):
        """Test max height calculation: h = vy0²/(2g)."""
        proj = ProjectileMotion(v0=50.0, angle=90.0)  # Vertical launch
        
        expected_height = (proj.vy0 ** 2) / (2 * G)
        actual_height = proj.max_height_analytical()
        
        assert np.isclose(actual_height, expected_height, rtol=1e-10)
    
    def test_range_at_45_degrees(self):
        """Range is maximized at 45 degrees."""
        v0 = 50.0
        proj_45 = ProjectileMotion(v0=v0, angle=45.0)
        proj_30 = ProjectileMotion(v0=v0, angle=30.0)
        proj_60 = ProjectileMotion(v0=v0, angle=60.0)
        
        range_45 = proj_45.range_analytical()
        range_30 = proj_30.range_analytical()
        range_60 = proj_60.range_analytical()
        
        assert range_45 > range_30
        assert range_45 > range_60
    
    def test_complementary_angles_same_range(self):
        """Complementary angles (30° and 60°) give same range."""
        v0 = 50.0
        range_30 = ProjectileMotion(v0=v0, angle=30.0).range_analytical()
        range_60 = ProjectileMotion(v0=v0, angle=60.0).range_analytical()
        
        assert np.isclose(range_30, range_60, rtol=1e-10)
    
    def test_time_of_flight_formula(self):
        """Test T = 2*vy0/g."""
        proj = ProjectileMotion(v0=50.0, angle=45.0)
        
        expected_time = 2 * proj.vy0 / G
        actual_time = proj.time_of_flight_analytical()
        
        assert np.isclose(actual_time, expected_time, rtol=1e-10)
    
    def test_range_formula(self):
        """Test R = v0² * sin(2θ) / g."""
        v0, angle = 50.0, 45.0
        proj = ProjectileMotion(v0=v0, angle=angle)
        
        expected_range = (v0 ** 2) * np.sin(2 * np.radians(angle)) / G
        actual_range = proj.range_analytical()
        
        assert np.isclose(actual_range, expected_range, rtol=1e-10)


class TestNumericalSolutions:
    """Test numerical integration methods."""
    
    def test_euler_trajectory_starts_at_origin(self):
        """Euler trajectory should start at initial position."""
        proj = ProjectileMotion(v0=50.0, angle=45.0)
        traj = proj.trajectory_euler(dt=0.01)
        
        assert traj[0].x == 0.0
        assert traj[0].y == 0.0
        assert traj[0].t == 0.0
    
    def test_rk4_trajectory_starts_at_origin(self):
        """RK4 trajectory should start at initial position."""
        proj = ProjectileMotion(v0=50.0, angle=45.0)
        traj = proj.trajectory_rk4(dt=0.01)
        
        assert traj[0].x == 0.0
        assert traj[0].y == 0.0
    
    def test_euler_converges_to_analytical(self):
        """Euler should converge to analytical as dt → 0."""
        proj = ProjectileMotion(v0=50.0, angle=45.0)
        
        analytical_range = proj.range_analytical()
        
        # Coarse step
        traj_coarse = proj.trajectory_euler(dt=0.1)
        range_coarse = traj_coarse[-1].x
        
        # Fine step
        traj_fine = proj.trajectory_euler(dt=0.001)
        range_fine = traj_fine[-1].x
        
        error_coarse = abs(range_coarse - analytical_range)
        error_fine = abs(range_fine - analytical_range)
        
        assert error_fine < error_coarse
    
    def test_rk4_more_accurate_than_euler(self):
        """RK4 should be more accurate than Euler for same step size."""
        proj = ProjectileMotion(v0=50.0, angle=45.0)
        dt = 0.05
        
        analytical_range = proj.range_analytical()
        euler_range = proj.trajectory_euler(dt)[-1].x
        rk4_range = proj.trajectory_rk4(dt)[-1].x
        
        euler_error = abs(euler_range - analytical_range)
        rk4_error = abs(rk4_range - analytical_range)
        
        assert rk4_error < euler_error
    
    def test_trajectory_ends_at_ground(self):
        """Trajectory should end near y=0."""
        proj = ProjectileMotion(v0=50.0, angle=45.0)
        traj = proj.trajectory_rk4(dt=0.01)
        
        # Last point should be at or below ground (allow for step size overshoot)
        assert traj[-1].y <= 0.5  # Allow reasonable overshoot based on dt


class TestComparisonMethods:
    """Test method comparison utilities."""
    
    def test_compare_methods_returns_all_fields(self):
        """Compare methods should return complete results."""
        proj = ProjectileMotion(v0=50.0, angle=45.0)
        results = proj.compare_methods(dt=0.01)
        
        assert 'analytical' in results
        assert 'euler' in results
        assert 'rk4' in results
        assert 'step_size' in results
        
        for method in ['analytical', 'euler', 'rk4']:
            assert 'range' in results[method]
            assert 'max_height' in results[method]
            assert 'time_of_flight' in results[method]


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_calculate_range(self):
        """Test calculate_range function."""
        range_val = calculate_range(v0=50.0, angle=45.0)
        expected = (50.0 ** 2) * np.sin(2 * np.radians(45)) / G
        
        assert np.isclose(range_val, expected)
    
    def test_calculate_max_height(self):
        """Test calculate_max_height function."""
        height = calculate_max_height(v0=50.0, angle=90.0)
        expected = (50.0 ** 2) / (2 * G)
        
        assert np.isclose(height, expected)
    
    def test_calculate_flight_time(self):
        """Test calculate_flight_time function."""
        v0, angle = 50.0, 45.0
        time = calculate_flight_time(v0=v0, angle=angle)
        vy0 = v0 * np.sin(np.radians(angle))
        expected = 2 * vy0 / G
        
        assert np.isclose(time, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])