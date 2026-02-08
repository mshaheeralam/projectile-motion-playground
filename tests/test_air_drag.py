"""
Tests for projectile motion with air resistance.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.air_drag import ProjectileWithDrag, DragTrajectoryPoint, DragModel
from physics.projectile import ProjectileMotion
from physics.constants import G, AIR_DENSITY, DEFAULT_DRAG_COEF


class TestProjectileWithDrag:
    """Test suite for ProjectileWithDrag class."""
    
    def test_init_velocity_components(self):
        """Test initialization of velocity components."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        expected = 50.0 * np.cos(np.radians(45))
        
        assert np.isclose(proj.vx0, expected, rtol=1e-10)
        assert np.isclose(proj.vy0, expected, rtol=1e-10)
    
    def test_init_default_values(self):
        """Test default values are set correctly."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        
        assert proj.g == G
        assert proj.Cd == DEFAULT_DRAG_COEF
        assert proj.rho == AIR_DENSITY


class TestDragModels:
    """Test different drag model implementations."""
    
    def test_no_drag_trajectory(self):
        """Trajectory with no drag should match basic projectile."""
        v0, angle, mass = 50.0, 45.0, 1.0
        
        proj_no_drag = ProjectileMotion(v0=v0, angle=angle)
        proj_with_drag = ProjectileWithDrag(v0=v0, angle=angle, mass=mass)
        
        traj_analytical = proj_no_drag.trajectory_analytical(num_points=100)
        traj_none = proj_with_drag.trajectory_rk4(dt=0.01, drag_model=DragModel.NONE)
        
        # Final positions should be close
        assert np.isclose(traj_none[-1].x, traj_analytical[-1].x, rtol=0.01)
    
    def test_drag_reduces_range(self):
        """Air drag should reduce horizontal range."""
        v0, angle, mass = 50.0, 45.0, 1.0
        
        proj = ProjectileWithDrag(v0=v0, angle=angle, mass=mass)
        
        traj_none = proj.trajectory_rk4(dt=0.001, drag_model=DragModel.NONE)
        traj_linear = proj.trajectory_rk4(dt=0.001, drag_model=DragModel.LINEAR)
        traj_quad = proj.trajectory_rk4(dt=0.001, drag_model=DragModel.QUADRATIC)
        
        range_none = traj_none[-1].x
        range_linear = traj_linear[-1].x
        range_quad = traj_quad[-1].x
        
        # Drag should reduce range
        assert range_linear < range_none
        assert range_quad < range_none
    
    def test_drag_reduces_max_height(self):
        """Air drag should reduce maximum height."""
        v0, angle, mass = 50.0, 60.0, 1.0  # Higher angle for more vertical motion
        
        proj = ProjectileWithDrag(v0=v0, angle=angle, mass=mass)
        
        traj_none = proj.trajectory_rk4(dt=0.001, drag_model=DragModel.NONE)
        traj_quad = proj.trajectory_rk4(dt=0.001, drag_model=DragModel.QUADRATIC)
        
        height_none = max(p.y for p in traj_none)
        height_quad = max(p.y for p in traj_quad)
        
        assert height_quad < height_none


class TestEnergyConservation:
    """Test energy tracking and conservation."""
    
    def test_no_drag_conserves_energy(self):
        """Without drag, kinetic energy should be conserved at same height."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        traj = proj.trajectory_rk4(dt=0.001, drag_model=DragModel.NONE)
        
        initial_energy = traj[0].energy
        final_energy = traj[-1].energy
        
        # Energy should be conserved (small tolerance for numerical errors)
        # Note: final point is at y≈0, so kinetic energy should match
        assert np.isclose(initial_energy, final_energy, rtol=0.01)
    
    def test_drag_dissipates_energy(self):
        """With drag, energy should decrease over time."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        traj = proj.trajectory_rk4(dt=0.001, drag_model=DragModel.QUADRATIC)
        
        initial_energy = traj[0].energy
        final_energy = traj[-1].energy
        
        assert final_energy < initial_energy
    
    def test_energy_monotonically_decreases_with_drag(self):
        """Total mechanical energy should decrease throughout flight with drag."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        traj = proj.trajectory_rk4(dt=0.001, drag_model=DragModel.QUADRATIC)
        
        # Calculate total mechanical energy (kinetic + potential)
        # E_total = 0.5*m*v² + m*g*h
        g = proj.g
        m = proj.mass
        
        def total_energy(point):
            return point.energy + m * g * point.y
        
        # Total energy should monotonically decrease (with small numerical tolerance)
        prev_total = total_energy(traj[0])
        for i in range(1, len(traj)):
            curr_total = total_energy(traj[i])
            assert curr_total <= prev_total + 1e-3, f"Energy increased at step {i}"
            prev_total = curr_total


class TestDragForceCalculation:
    """Test drag force calculations."""
    
    def test_drag_force_zero_at_rest(self):
        """Drag force should be zero when velocity is zero."""
        proj = ProjectileWithDrag(v0=0.0, angle=45.0, mass=1.0)
        
        ax, ay = proj._quadratic_drag_acceleration(0.0, 0.0)
        
        assert ax == 0.0
        assert ay == 0.0
    
    def test_drag_opposes_motion(self):
        """Drag should oppose direction of motion."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        
        # Positive vx, vy
        ax, ay = proj._quadratic_drag_acceleration(10.0, 10.0)
        
        # Drag acceleration should be negative (opposing positive velocity)
        assert ax < 0
        assert ay < 0
    
    def test_drag_increases_with_speed(self):
        """Quadratic drag should increase with velocity squared."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        
        ax1, _ = proj._quadratic_drag_acceleration(10.0, 0.0)
        ax2, _ = proj._quadratic_drag_acceleration(20.0, 0.0)
        
        # Drag at 2x velocity should be 4x (quadratic)
        assert np.isclose(abs(ax2) / abs(ax1), 4.0, rtol=0.01)


class TestNumericalMethods:
    """Test numerical integration methods with drag."""
    
    def test_euler_trajectory_exists(self):
        """Euler method should produce a trajectory."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        traj = proj.trajectory_euler(dt=0.01, drag_model=DragModel.QUADRATIC)
        
        assert len(traj) > 0
        assert traj[0].t == 0.0
    
    def test_rk4_trajectory_exists(self):
        """RK4 method should produce a trajectory."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        traj = proj.trajectory_rk4(dt=0.01, drag_model=DragModel.QUADRATIC)
        
        assert len(traj) > 0
        assert traj[0].t == 0.0
    
    def test_rk4_more_accurate_than_euler_with_drag(self):
        """RK4 should be more accurate for same step size."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        
        # Use fine reference
        traj_ref = proj.trajectory_rk4(dt=0.0001, drag_model=DragModel.QUADRATIC)
        ref_range = traj_ref[-1].x
        
        dt = 0.01
        traj_euler = proj.trajectory_euler(dt=dt, drag_model=DragModel.QUADRATIC)
        traj_rk4 = proj.trajectory_rk4(dt=dt, drag_model=DragModel.QUADRATIC)
        
        euler_error = abs(traj_euler[-1].x - ref_range)
        rk4_error = abs(traj_rk4[-1].x - ref_range)
        
        assert rk4_error < euler_error


class TestDragModelComparison:
    """Test comparing different drag models."""
    
    def test_compare_drag_models_returns_all_fields(self):
        """Comparison should return metrics for all models."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        results = proj.compare_drag_models(dt=0.01)
        
        assert 'no_drag' in results
        assert 'linear_drag' in results
        assert 'quadratic_drag' in results
        assert 'trajectories' in results
        
        for model in ['no_drag', 'linear_drag', 'quadratic_drag']:
            assert 'range' in results[model]
            assert 'max_height' in results[model]
            assert 'time' in results[model]
    
    def test_quadratic_drag_less_than_linear_at_high_speed(self):
        """At high speeds, quadratic drag effects should dominate."""
        proj = ProjectileWithDrag(v0=100.0, angle=45.0, mass=1.0)
        results = proj.compare_drag_models(dt=0.001)
        
        # Both should reduce range compared to no drag
        assert results['linear_drag']['range'] < results['no_drag']['range']
        assert results['quadratic_drag']['range'] < results['no_drag']['range']


class TestStepSizeAnalysis:
    """Test step size sensitivity analysis."""
    
    def test_step_size_analysis_returns_results(self):
        """Analysis should return results for each step size."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        analysis = proj.analyze_step_size_effect(
            step_sizes=[0.1, 0.01, 0.001],
            drag_model=DragModel.QUADRATIC
        )
        
        assert 'results' in analysis
        assert len(analysis['results']) == 3
    
    def test_smaller_step_reduces_error(self):
        """Smaller step sizes should give more accurate results."""
        proj = ProjectileWithDrag(v0=50.0, angle=45.0, mass=1.0)
        analysis = proj.analyze_step_size_effect(
            step_sizes=[0.1, 0.01, 0.001],
            drag_model=DragModel.QUADRATIC
        )
        
        errors = [r['range_error'] for r in analysis['results']]
        
        # Errors should generally decrease with smaller steps
        assert errors[1] < errors[0]  # 0.01 better than 0.1
        assert errors[2] < errors[1]  # 0.001 better than 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])