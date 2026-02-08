"""
Projectile motion with air resistance.
Implements linear and quadratic drag models using numerical integration.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from .constants import G, AIR_DENSITY, DEFAULT_DRAG_COEF, cross_sectional_area


class DragModel(Enum):
    """Types of drag models."""
    NONE = "none"
    LINEAR = "linear"
    QUADRATIC = "quadratic"


@dataclass
class DragTrajectoryPoint:
    """Trajectory point with drag-related data."""
    t: float      # Time (s)
    x: float      # Horizontal position (m)
    y: float      # Vertical position (m)
    vx: float     # Horizontal velocity (m/s)
    vy: float     # Vertical velocity (m/s)
    speed: float  # Total speed (m/s)
    drag: float   # Drag force magnitude (N)
    energy: float # Kinetic energy (J)


class ProjectileWithDrag:
    """
    Projectile motion with air resistance.
    
    Linear drag: F_drag = -b * v (Stokes drag, low Reynolds number)
    Quadratic drag: F_drag = -0.5 * ρ * Cd * A * v² (turbulent, high Re)
    
    No analytical solution exists - must use numerical methods.
    """
    
    def __init__(self, v0: float, angle: float, mass: float,
                 x0: float = 0.0, y0: float = 0.0, g: float = G,
                 drag_coefficient: float = DEFAULT_DRAG_COEF,
                 radius: float = 0.05,
                 air_density: float = AIR_DENSITY,
                 linear_drag_coef: Optional[float] = None):
        """
        Initialize projectile with drag.
        
        Args:
            v0: Initial velocity magnitude (m/s)
            angle: Launch angle from horizontal (degrees)
            mass: Projectile mass (kg)
            x0: Initial x position (m)
            y0: Initial y position (m)
            g: Gravitational acceleration (m/s²)
            drag_coefficient: Drag coefficient Cd (dimensionless)
            radius: Projectile radius for area calculation (m)
            air_density: Air density ρ (kg/m³)
            linear_drag_coef: Linear drag coefficient b (kg/s), if using linear model
        """
        self.v0 = v0
        self.angle_deg = angle
        self.angle_rad = np.radians(angle)
        self.mass = mass
        self.x0 = x0
        self.y0 = y0
        self.g = g
        self.Cd = drag_coefficient
        self.radius = radius
        self.area = cross_sectional_area(radius)
        self.rho = air_density
        self.b = linear_drag_coef if linear_drag_coef else 0.1 * mass  # Default linear coef
        
        # Initial velocity components
        self.vx0 = v0 * np.cos(self.angle_rad)
        self.vy0 = v0 * np.sin(self.angle_rad)
    
    def _linear_drag_acceleration(self, vx: float, vy: float) -> Tuple[float, float]:
        """
        Calculate acceleration due to linear drag.
        
        F = -b * v → a = -b/m * v
        """
        ax = -(self.b / self.mass) * vx
        ay = -(self.b / self.mass) * vy
        return ax, ay
    
    def _quadratic_drag_acceleration(self, vx: float, vy: float) -> Tuple[float, float]:
        """
        Calculate acceleration due to quadratic drag.
        
        F = 0.5 * ρ * Cd * A * v² in direction opposite to velocity
        """
        speed = np.sqrt(vx**2 + vy**2)
        if speed < 1e-10:
            return 0.0, 0.0
        
        # Drag force magnitude
        F_drag = 0.5 * self.rho * self.Cd * self.area * speed**2
        
        # Acceleration components (opposite to velocity direction)
        ax = -(F_drag / self.mass) * (vx / speed)
        ay = -(F_drag / self.mass) * (vy / speed)
        return ax, ay
    
    def trajectory_rk4(self, dt: float = 0.001, 
                       drag_model: DragModel = DragModel.QUADRATIC,
                       max_time: float = 100.0) -> List[DragTrajectoryPoint]:
        """
        Solve trajectory with drag using RK4.
        
        Args:
            dt: Time step (s) - smaller needed for drag accuracy
            drag_model: Type of drag to apply
            max_time: Maximum simulation time (s)
        """
        trajectory = []
        t, x, y = 0.0, self.x0, self.y0
        vx, vy = self.vx0, self.vy0
        
        def get_drag_accel(vx_s: float, vy_s: float) -> Tuple[float, float]:
            if drag_model == DragModel.NONE:
                return 0.0, 0.0
            elif drag_model == DragModel.LINEAR:
                return self._linear_drag_acceleration(vx_s, vy_s)
            else:  # QUADRATIC
                return self._quadratic_drag_acceleration(vx_s, vy_s)
        
        def derivatives(state: np.ndarray) -> np.ndarray:
            """Return [dx/dt, dy/dt, dvx/dt, dvy/dt]"""
            _, _, vx_s, vy_s = state
            ax_drag, ay_drag = get_drag_accel(vx_s, vy_s)
            return np.array([vx_s, vy_s, ax_drag, -self.g + ay_drag])
        
        while y >= 0 and t < max_time:
            speed = np.sqrt(vx**2 + vy**2)
            
            # Calculate current drag force
            if drag_model == DragModel.LINEAR:
                drag_force = self.b * speed
            elif drag_model == DragModel.QUADRATIC:
                drag_force = 0.5 * self.rho * self.Cd * self.area * speed**2
            else:
                drag_force = 0.0
            
            kinetic_energy = 0.5 * self.mass * speed**2
            
            trajectory.append(DragTrajectoryPoint(
                t=t, x=x, y=y, vx=vx, vy=vy,
                speed=speed, drag=drag_force, energy=kinetic_energy
            ))
            
            # RK4 integration
            state = np.array([x, y, vx, vy])
            k1 = derivatives(state)
            k2 = derivatives(state + 0.5 * dt * k1)
            k3 = derivatives(state + 0.5 * dt * k2)
            k4 = derivatives(state + dt * k3)
            
            state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x, y, vx, vy = state
            t += dt
        
        return trajectory
    
    def trajectory_euler(self, dt: float = 0.001,
                         drag_model: DragModel = DragModel.QUADRATIC,
                         max_time: float = 100.0) -> List[DragTrajectoryPoint]:
        """
        Solve trajectory with drag using Euler's method.
        Less accurate than RK4, useful for comparison.
        """
        trajectory = []
        t, x, y = 0.0, self.x0, self.y0
        vx, vy = self.vx0, self.vy0
        
        while y >= 0 and t < max_time:
            speed = np.sqrt(vx**2 + vy**2)
            
            # Calculate drag
            if drag_model == DragModel.LINEAR:
                ax_drag, ay_drag = self._linear_drag_acceleration(vx, vy)
                drag_force = self.b * speed
            elif drag_model == DragModel.QUADRATIC:
                ax_drag, ay_drag = self._quadratic_drag_acceleration(vx, vy)
                drag_force = 0.5 * self.rho * self.Cd * self.area * speed**2
            else:
                ax_drag, ay_drag = 0.0, 0.0
                drag_force = 0.0
            
            kinetic_energy = 0.5 * self.mass * speed**2
            
            trajectory.append(DragTrajectoryPoint(
                t=t, x=x, y=y, vx=vx, vy=vy,
                speed=speed, drag=drag_force, energy=kinetic_energy
            ))
            
            # Update velocity
            vx += ax_drag * dt
            vy += (-self.g + ay_drag) * dt
            
            # Update position
            x += vx * dt
            y += vy * dt
            
            t += dt
        
        return trajectory
    
    def compare_drag_models(self, dt: float = 0.001) -> Dict:
        """
        Compare trajectories with no drag, linear drag, and quadratic drag.
        """
        traj_none = self.trajectory_rk4(dt, DragModel.NONE)
        traj_linear = self.trajectory_rk4(dt, DragModel.LINEAR)
        traj_quadratic = self.trajectory_rk4(dt, DragModel.QUADRATIC)
        
        def get_metrics(traj: List[DragTrajectoryPoint]) -> Dict:
            if not traj:
                return {'range': 0, 'time': 0, 'max_height': 0, 'energy_loss': 0}
            return {
                'range': traj[-1].x,
                'time': traj[-1].t,
                'max_height': max(p.y for p in traj),
                'final_energy': traj[-1].energy,
                'initial_energy': traj[0].energy,
                'energy_loss_pct': 100 * (1 - traj[-1].energy / traj[0].energy) if traj[0].energy > 0 else 0,
            }
        
        return {
            'no_drag': get_metrics(traj_none),
            'linear_drag': get_metrics(traj_linear),
            'quadratic_drag': get_metrics(traj_quadratic),
            'trajectories': {
                'no_drag': traj_none,
                'linear_drag': traj_linear,
                'quadratic_drag': traj_quadratic,
            }
        }

    def analyze_step_size_effect(self, step_sizes: List[float] = None,
                                  drag_model: DragModel = DragModel.QUADRATIC) -> Dict:
        """
        Analyze how step size affects accuracy and performance.
        Useful for understanding numerical stability.
        """
        if step_sizes is None:
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
        
        results = []
        # Use smallest step size as reference
        ref_traj = self.trajectory_rk4(min(step_sizes), drag_model)
        ref_range = ref_traj[-1].x if ref_traj else 0
        ref_time = ref_traj[-1].t if ref_traj else 0
        
        import time as time_module
        
        for dt in step_sizes:
            start = time_module.perf_counter()
            traj = self.trajectory_rk4(dt, drag_model)
            elapsed = time_module.perf_counter() - start
            
            if traj:
                results.append({
                    'step_size': dt,
                    'range': traj[-1].x,
                    'time_of_flight': traj[-1].t,
                    'range_error': abs(traj[-1].x - ref_range),
                    'time_error': abs(traj[-1].t - ref_time),
                    'num_steps': len(traj),
                    'computation_time_ms': elapsed * 1000,
                })
        
        return {
            'reference_step_size': min(step_sizes),
            'results': results
        }