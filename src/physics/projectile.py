"""
Projectile motion physics without air resistance.
Implements both analytical (closed-form) and numerical solutions.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from .constants import G


@dataclass
class TrajectoryPoint:
    """Represents a point in the trajectory."""
    t: float  # Time (s)
    x: float  # Horizontal position (m)
    y: float  # Vertical position (m)
    vx: float  # Horizontal velocity (m/s)
    vy: float  # Vertical velocity (m/s)


class ProjectileMotion:
    """
    Classical projectile motion without air resistance.
    Provides both analytical and numerical solutions for comparison.
    """
    
    def __init__(self, v0: float, angle: float, x0: float = 0.0, y0: float = 0.0, 
                 g: float = G):
        """
        Initialize projectile motion.
        
        Args:
            v0: Initial velocity magnitude (m/s)
            angle: Launch angle from horizontal (degrees)
            x0: Initial x position (m)
            y0: Initial y position (m)
            g: Gravitational acceleration (m/s²)
        """
        self.v0 = v0
        self.angle_deg = angle
        self.angle_rad = np.radians(angle)
        self.x0 = x0
        self.y0 = y0
        self.g = g
        
        # Initial velocity components
        self.vx0 = v0 * np.cos(self.angle_rad)
        self.vy0 = v0 * np.sin(self.angle_rad)
    
    # ==================== ANALYTICAL SOLUTIONS ====================
    
    def position_analytical(self, t: float) -> Tuple[float, float]:
        """
        Calculate position at time t using closed-form equations.
        
        x(t) = x0 + vx0 * t
        y(t) = y0 + vy0 * t - 0.5 * g * t²
        """
        x = self.x0 + self.vx0 * t
        y = self.y0 + self.vy0 * t - 0.5 * self.g * t**2
        return x, y
    
    def velocity_analytical(self, t: float) -> Tuple[float, float]:
        """
        Calculate velocity at time t using closed-form equations.
        
        vx(t) = vx0 (constant)
        vy(t) = vy0 - g * t
        """
        vx = self.vx0
        vy = self.vy0 - self.g * t
        return vx, vy
    
    def time_of_flight_analytical(self) -> float:
        """
        Calculate total flight time until projectile returns to launch height.
        
        T = 2 * vy0 / g
        """
        if self.y0 == 0:
            return 2 * self.vy0 / self.g
        else:
            # Solve y0 + vy0*t - 0.5*g*t² = 0 for t > 0
            discriminant = self.vy0**2 + 2 * self.g * self.y0
            if discriminant < 0:
                return 0.0  # Never lands
            return (self.vy0 + np.sqrt(discriminant)) / self.g
    
    def max_height_analytical(self) -> float:
        """
        Calculate maximum height reached.
        
        h_max = y0 + vy0² / (2g)
        """
        return self.y0 + (self.vy0**2) / (2 * self.g)
    
    def range_analytical(self) -> float:
        """
        Calculate horizontal range (distance when y returns to y0).
        
        R = vx0 * T = v0² * sin(2θ) / g  (when y0 = 0)
        """
        t_flight = self.time_of_flight_analytical()
        return self.vx0 * t_flight
    
    def trajectory_analytical(self, num_points: int = 100) -> List[TrajectoryPoint]:
        """Generate full trajectory using analytical solution."""
        t_flight = self.time_of_flight_analytical()
        times = np.linspace(0, t_flight, num_points)
        
        trajectory = []
        for t in times:
            x, y = self.position_analytical(t)
            vx, vy = self.velocity_analytical(t)
            trajectory.append(TrajectoryPoint(t=t, x=x, y=y, vx=vx, vy=vy))
        
        return trajectory
    
    # ==================== NUMERICAL SOLUTIONS ====================
    
    def trajectory_euler(self, dt: float = 0.01) -> List[TrajectoryPoint]:
        """
        Solve trajectory using Euler's method (first-order).
        
        x_{n+1} = x_n + vx_n * dt
        y_{n+1} = y_n + vy_n * dt
        vx_{n+1} = vx_n
        vy_{n+1} = vy_n - g * dt
        """
        trajectory = []
        t, x, y = 0.0, self.x0, self.y0
        vx, vy = self.vx0, self.vy0
        
        while y >= 0:
            trajectory.append(TrajectoryPoint(t=t, x=x, y=y, vx=vx, vy=vy))
            
            # Update position
            x += vx * dt
            y += vy * dt
            
            # Update velocity
            vy -= self.g * dt
            
            t += dt
        
        return trajectory
    
    def trajectory_rk4(self, dt: float = 0.01) -> List[TrajectoryPoint]:
        """
        Solve trajectory using 4th-order Runge-Kutta method.
        More accurate than Euler for the same step size.
        
        For projectile without drag, accelerations are:
        ax = 0
        ay = -g
        """
        trajectory = []
        t, x, y = 0.0, self.x0, self.y0
        vx, vy = self.vx0, self.vy0
        
        def derivatives(state):
            """Return (dx/dt, dy/dt, dvx/dt, dvy/dt)"""
            _, _, vx_s, vy_s = state
            return np.array([vx_s, vy_s, 0.0, -self.g])
        
        while y >= 0:
            trajectory.append(TrajectoryPoint(t=t, x=x, y=y, vx=vx, vy=vy))
            
            state = np.array([x, y, vx, vy])
            
            # RK4 steps
            k1 = derivatives(state)
            k2 = derivatives(state + 0.5 * dt * k1)
            k3 = derivatives(state + 0.5 * dt * k2)
            k4 = derivatives(state + dt * k3)
            
            # Update state
            state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x, y, vx, vy = state
            t += dt
        
        return trajectory
    
    def compare_methods(self, dt: float = 0.01) -> Dict:
        """
        Compare analytical, Euler, and RK4 solutions.
        
        Returns metrics: time_of_flight, range, max_height, and errors.
        """
        # Analytical results (ground truth)
        t_analytical = self.time_of_flight_analytical()
        range_analytical = self.range_analytical()
        height_analytical = self.max_height_analytical()
        
        # Euler results
        traj_euler = self.trajectory_euler(dt)
        t_euler = traj_euler[-1].t if traj_euler else 0
        range_euler = traj_euler[-1].x if traj_euler else 0
        height_euler = max(p.y for p in traj_euler) if traj_euler else 0
        
        # RK4 results
        traj_rk4 = self.trajectory_rk4(dt)
        t_rk4 = traj_rk4[-1].t if traj_rk4 else 0
        range_rk4 = traj_rk4[-1].x if traj_rk4 else 0
        height_rk4 = max(p.y for p in traj_rk4) if traj_rk4 else 0
        
        return {
            'analytical': {
                'time_of_flight': t_analytical,
                'range': range_analytical,
                'max_height': height_analytical,
            },
            'euler': {
                'time_of_flight': t_euler,
                'range': range_euler,
                'max_height': height_euler,
                'time_error': abs(t_euler - t_analytical),
                'range_error': abs(range_euler - range_analytical),
                'height_error': abs(height_euler - height_analytical),
            },
            'rk4': {
                'time_of_flight': t_rk4,
                'range': range_rk4,
                'max_height': height_rk4,
                'time_error': abs(t_rk4 - t_analytical),
                'range_error': abs(range_rk4 - range_analytical),
                'height_error': abs(height_rk4 - height_analytical),
            },
            'step_size': dt,
        }


# Convenience functions for simple use cases
def calculate_range(v0: float, angle: float, g: float = G) -> float:
    """Quick calculation of projectile range."""
    return ProjectileMotion(v0, angle, g=g).range_analytical()


def calculate_max_height(v0: float, angle: float, g: float = G) -> float:
    """Quick calculation of maximum height."""
    return ProjectileMotion(v0, angle, g=g).max_height_analytical()


def calculate_flight_time(v0: float, angle: float, g: float = G) -> float:
    """Quick calculation of flight time."""
    return ProjectileMotion(v0, angle, g=g).time_of_flight_analytical()