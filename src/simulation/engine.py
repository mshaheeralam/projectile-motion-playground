"""
Simulation engine for projectile motion.
Orchestrates physics calculations and provides unified interface.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time as time_module

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics.projectile import ProjectileMotion, TrajectoryPoint
from physics.air_drag import ProjectileWithDrag, DragTrajectoryPoint, DragModel
from physics.constants import G, AIR_DENSITY, DEFAULT_DRAG_COEF


class SolutionMethod(Enum):
    """Available solution methods."""
    ANALYTICAL = "analytical"
    EULER = "euler"
    RK4 = "rk4"


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    # Initial conditions
    v0: float = 50.0          # Initial velocity (m/s)
    angle: float = 45.0        # Launch angle (degrees)
    x0: float = 0.0            # Initial x position (m)
    y0: float = 0.0            # Initial y position (m)
    
    # Physical properties
    mass: float = 1.0          # Mass (kg)
    g: float = G               # Gravity (m/s²)
    
    # Drag properties
    drag_model: DragModel = DragModel.NONE
    drag_coefficient: float = DEFAULT_DRAG_COEF
    radius: float = 0.05       # Projectile radius (m)
    air_density: float = AIR_DENSITY
    
    # Numerical parameters
    dt: float = 0.01           # Time step (s)
    method: SolutionMethod = SolutionMethod.RK4


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    config: SimulationConfig
    trajectory: List[Any]  # List of TrajectoryPoint or DragTrajectoryPoint
    metrics: Dict = field(default_factory=dict)
    computation_time_ms: float = 0.0
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert trajectory to numpy arrays for plotting."""
        if not self.trajectory:
            return {'t': np.array([]), 'x': np.array([]), 'y': np.array([]),
                    'vx': np.array([]), 'vy': np.array([])}
        
        return {
            't': np.array([p.t for p in self.trajectory]),
            'x': np.array([p.x for p in self.trajectory]),
            'y': np.array([p.y for p in self.trajectory]),
            'vx': np.array([p.vx for p in self.trajectory]),
            'vy': np.array([p.vy for p in self.trajectory]),
        }


class SimulationEngine:
    """
    Main simulation engine.
    Provides unified interface for running various projectile simulations.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize engine with optional configuration."""
        self.config = config or SimulationConfig()
        self._results_cache: Dict[str, SimulationResult] = {}
    
    def run(self, config: Optional[SimulationConfig] = None) -> SimulationResult:
        """
        Run a single simulation with given configuration.
        """
        cfg = config or self.config
        start_time = time_module.perf_counter()
        
        if cfg.drag_model == DragModel.NONE:
            # Use basic projectile motion (has analytical solution)
            projectile = ProjectileMotion(
                v0=cfg.v0, angle=cfg.angle, x0=cfg.x0, y0=cfg.y0, g=cfg.g
            )
            
            if cfg.method == SolutionMethod.ANALYTICAL:
                trajectory = projectile.trajectory_analytical()
            elif cfg.method == SolutionMethod.EULER:
                trajectory = projectile.trajectory_euler(dt=cfg.dt)
            else:  # RK4
                trajectory = projectile.trajectory_rk4(dt=cfg.dt)
            
            metrics = {
                'range': trajectory[-1].x if trajectory else 0,
                'max_height': max(p.y for p in trajectory) if trajectory else 0,
                'time_of_flight': trajectory[-1].t if trajectory else 0,
                'final_speed': np.sqrt(trajectory[-1].vx**2 + trajectory[-1].vy**2) if trajectory else 0,
            }
            
            # Add analytical reference for comparison
            if cfg.method != SolutionMethod.ANALYTICAL:
                metrics['analytical_range'] = projectile.range_analytical()
                metrics['analytical_height'] = projectile.max_height_analytical()
                metrics['analytical_time'] = projectile.time_of_flight_analytical()
                metrics['range_error'] = abs(metrics['range'] - metrics['analytical_range'])
                metrics['height_error'] = abs(metrics['max_height'] - metrics['analytical_height'])
        
        else:
            # Use projectile with drag (numerical only)
            projectile = ProjectileWithDrag(
                v0=cfg.v0, angle=cfg.angle, mass=cfg.mass,
                x0=cfg.x0, y0=cfg.y0, g=cfg.g,
                drag_coefficient=cfg.drag_coefficient,
                radius=cfg.radius, air_density=cfg.air_density
            )
            
            if cfg.method == SolutionMethod.EULER:
                trajectory = projectile.trajectory_euler(dt=cfg.dt, drag_model=cfg.drag_model)
            else:  # RK4 or ANALYTICAL (use RK4 for drag)
                trajectory = projectile.trajectory_rk4(dt=cfg.dt, drag_model=cfg.drag_model)
            
            if trajectory:
                metrics = {
                    'range': trajectory[-1].x,
                    'max_height': max(p.y for p in trajectory),
                    'time_of_flight': trajectory[-1].t,
                    'final_speed': trajectory[-1].speed,
                    'initial_energy': trajectory[0].energy,
                    'final_energy': trajectory[-1].energy,
                    'energy_loss_pct': 100 * (1 - trajectory[-1].energy / trajectory[0].energy),
                    'max_drag_force': max(p.drag for p in trajectory),
                }
            else:
                metrics = {}
        
        elapsed = time_module.perf_counter() - start_time
        
        return SimulationResult(
            config=cfg,
            trajectory=trajectory,
            metrics=metrics,
            computation_time_ms=elapsed * 1000
        )
    
    def compare_methods(self, v0: float = 50.0, angle: float = 45.0,
                        dt: float = 0.01) -> Dict[str, SimulationResult]:
        """
        Compare different solution methods for the same initial conditions.
        """
        base_config = SimulationConfig(v0=v0, angle=angle, dt=dt, drag_model=DragModel.NONE)
        
        results = {}
        for method in SolutionMethod:
            cfg = SimulationConfig(
                v0=v0, angle=angle, dt=dt,
                drag_model=DragModel.NONE,
                method=method
            )
            results[method.value] = self.run(cfg)
        
        return results
    
    def compare_drag_models(self, v0: float = 50.0, angle: float = 45.0,
                            mass: float = 1.0, dt: float = 0.001) -> Dict[str, SimulationResult]:
        """
        Compare trajectories with different drag models.
        """
        results = {}
        for drag_model in DragModel:
            cfg = SimulationConfig(
                v0=v0, angle=angle, mass=mass, dt=dt,
                drag_model=drag_model,
                method=SolutionMethod.RK4
            )
            results[drag_model.value] = self.run(cfg)
        
        return results
    
    def parameter_sweep(self, param_name: str, values: List[float],
                        base_config: Optional[SimulationConfig] = None) -> List[SimulationResult]:
        """
        Run simulations sweeping over a parameter.
        Useful for sensitivity analysis.
        """
        base = base_config or self.config
        results = []
        
        for value in values:
            cfg_dict = {
                'v0': base.v0, 'angle': base.angle, 'mass': base.mass,
                'g': base.g, 'drag_model': base.drag_model, 'dt': base.dt,
                'drag_coefficient': base.drag_coefficient, 'radius': base.radius,
                'method': base.method
            }
            cfg_dict[param_name] = value
            cfg = SimulationConfig(**cfg_dict)
            results.append(self.run(cfg))
        
        return results
    
    def convergence_study(self, step_sizes: List[float] = None,
                          config: Optional[SimulationConfig] = None) -> Dict:
        """
        Study convergence behavior with different step sizes.
        """
        if step_sizes is None:
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        cfg = config or self.config
        results = []
        
        # Get reference (smallest step size)
        ref_cfg = SimulationConfig(**{**cfg.__dict__, 'dt': min(step_sizes)})
        ref_result = self.run(ref_cfg)
        ref_range = ref_result.metrics.get('range', 0)
        
        for dt in step_sizes:
            test_cfg = SimulationConfig(**{**cfg.__dict__, 'dt': dt})
            result = self.run(test_cfg)
            
            results.append({
                'dt': dt,
                'range': result.metrics.get('range', 0),
                'range_error': abs(result.metrics.get('range', 0) - ref_range),
                'computation_time_ms': result.computation_time_ms,
                'num_points': len(result.trajectory)
            })
        
        return {
            'reference_dt': min(step_sizes),
            'reference_range': ref_range,
            'results': results
        }


def quick_simulation(v0: float, angle: float, drag: bool = False,
                     mass: float = 1.0) -> SimulationResult:
    """
    Convenience function for quick simulations.
    """
    engine = SimulationEngine()
    config = SimulationConfig(
        v0=v0, angle=angle, mass=mass,
        drag_model=DragModel.QUADRATIC if drag else DragModel.NONE,
        method=SolutionMethod.ANALYTICAL if not drag else SolutionMethod.RK4
    )
    return engine.run(config)