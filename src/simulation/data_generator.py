"""
Data generator for ML training datasets.
Generates projectile motion data with various parameters.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics.projectile import ProjectileMotion
from physics.air_drag import ProjectileWithDrag, DragModel
from physics.constants import G, AIR_DENSITY, DRAG_COEFFICIENTS


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    num_samples: int = 10000
    
    # Parameter ranges
    v0_range: Tuple[float, float] = (10.0, 100.0)      # m/s
    angle_range: Tuple[float, float] = (5.0, 85.0)     # degrees
    mass_range: Tuple[float, float] = (0.1, 10.0)      # kg
    drag_coef_range: Tuple[float, float] = (0.2, 1.0)  # dimensionless
    radius_range: Tuple[float, float] = (0.01, 0.1)    # m
    
    # Fixed parameters (None = use from range)
    fixed_mass: Optional[float] = None
    fixed_drag_coef: Optional[float] = None
    
    # Include drag simulations
    include_no_drag: bool = True
    include_linear_drag: bool = True
    include_quadratic_drag: bool = True
    
    # Simulation settings
    dt: float = 0.001
    
    # Random seed for reproducibility
    seed: Optional[int] = 42


class DatasetGenerator:
    """
    Generates datasets for ML training.
    
    Features (inputs):
    - v0: initial velocity
    - angle: launch angle  
    - mass: projectile mass
    - drag_coef: drag coefficient
    - radius: projectile radius
    
    Labels (outputs):
    - range: horizontal distance traveled
    - max_height: maximum height reached
    - time_of_flight: total flight time
    - final_speed: impact speed
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
    
    def _sample_parameters(self) -> Dict:
        """Sample random parameters within configured ranges."""
        cfg = self.config
        
        return {
            'v0': np.random.uniform(*cfg.v0_range),
            'angle': np.random.uniform(*cfg.angle_range),
            'mass': cfg.fixed_mass or np.random.uniform(*cfg.mass_range),
            'drag_coef': cfg.fixed_drag_coef or np.random.uniform(*cfg.drag_coef_range),
            'radius': np.random.uniform(*cfg.radius_range),
        }
    
    def _simulate_no_drag(self, params: Dict) -> Dict:
        """Simulate without drag (uses analytical solution)."""
        proj = ProjectileMotion(v0=params['v0'], angle=params['angle'])
        
        return {
            'range_no_drag': proj.range_analytical(),
            'max_height_no_drag': proj.max_height_analytical(),
            'time_of_flight_no_drag': proj.time_of_flight_analytical(),
            'final_speed_no_drag': params['v0'],  # Same as initial for no drag from same height
        }
    
    def _simulate_with_drag(self, params: Dict, drag_model: DragModel) -> Dict:
        """Simulate with specified drag model."""
        proj = ProjectileWithDrag(
            v0=params['v0'],
            angle=params['angle'],
            mass=params['mass'],
            drag_coefficient=params['drag_coef'],
            radius=params['radius']
        )
        
        trajectory = proj.trajectory_rk4(dt=self.config.dt, drag_model=drag_model)
        
        suffix = '_linear_drag' if drag_model == DragModel.LINEAR else '_quad_drag'
        
        if trajectory:
            return {
                f'range{suffix}': trajectory[-1].x,
                f'max_height{suffix}': max(p.y for p in trajectory),
                f'time_of_flight{suffix}': trajectory[-1].t,
                f'final_speed{suffix}': trajectory[-1].speed,
                f'energy_loss_pct{suffix}': 100 * (1 - trajectory[-1].energy / trajectory[0].energy),
            }
        else:
            return {
                f'range{suffix}': 0,
                f'max_height{suffix}': 0,
                f'time_of_flight{suffix}': 0,
                f'final_speed{suffix}': 0,
                f'energy_loss_pct{suffix}': 0,
            }
    
    def generate(self, num_samples: Optional[int] = None, 
                 show_progress: bool = True) -> pd.DataFrame:
        """
        Generate dataset with specified number of samples.
        
        Returns DataFrame with features and labels.
        """
        n = num_samples or self.config.num_samples
        cfg = self.config
        
        data = []
        
        for i in range(n):
            if show_progress and (i + 1) % (n // 10 or 1) == 0:
                print(f"Generating sample {i + 1}/{n}...")
            
            # Sample parameters
            params = self._sample_parameters()
            
            # Start with input features
            row = {
                'v0': params['v0'],
                'angle': params['angle'],
                'mass': params['mass'],
                'drag_coef': params['drag_coef'],
                'radius': params['radius'],
                'area': np.pi * params['radius']**2,
            }
            
            # Add no-drag simulation results
            if cfg.include_no_drag:
                row.update(self._simulate_no_drag(params))
            
            # Add linear drag simulation results
            if cfg.include_linear_drag:
                row.update(self._simulate_with_drag(params, DragModel.LINEAR))
            
            # Add quadratic drag simulation results
            if cfg.include_quadratic_drag:
                row.update(self._simulate_with_drag(params, DragModel.QUADRATIC))
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_comparison_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate dataset specifically for comparing physics vs ML predictions.
        Includes ratio columns for analysis.
        """
        df = self.generate(num_samples, show_progress=True)
        
        # Add ratio columns (drag effect magnitude)
        if 'range_no_drag' in df.columns and 'range_quad_drag' in df.columns:
            df['range_reduction_pct'] = 100 * (1 - df['range_quad_drag'] / df['range_no_drag'])
            df['height_reduction_pct'] = 100 * (1 - df['max_height_quad_drag'] / df['max_height_no_drag'])
            df['time_reduction_pct'] = 100 * (1 - df['time_of_flight_quad_drag'] / df['time_of_flight_no_drag'])
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str, 
                     format: str = 'csv') -> None:
        """Save dataset to file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load dataset from file."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")


def generate_training_data(num_samples: int = 10000, 
                           output_path: str = None,
                           seed: int = 42) -> pd.DataFrame:
    """
    Convenience function to generate training data.
    """
    config = DatasetConfig(
        num_samples=num_samples,
        seed=seed,
        include_no_drag=True,
        include_linear_drag=False,  # Skip linear for main training
        include_quadratic_drag=True,
    )
    
    generator = DatasetGenerator(config)
    df = generator.generate()
    
    if output_path:
        generator.save_dataset(df, output_path)
    
    return df


def generate_test_cases(num_cases: int = 100) -> pd.DataFrame:
    """
    Generate specific test cases with known analytical solutions.
    Useful for validating ML models.
    """
    # Fixed parameters for analytical verification
    test_cases = []
    
    # Standard angles
    angles = [15, 30, 45, 60, 75]
    velocities = [20, 40, 60, 80, 100]
    
    for angle in angles:
        for v0 in velocities:
            proj = ProjectileMotion(v0=v0, angle=angle)
            test_cases.append({
                'v0': v0,
                'angle': angle,
                'mass': 1.0,
                'drag_coef': 0.0,
                'radius': 0.05,
                'expected_range': proj.range_analytical(),
                'expected_height': proj.max_height_analytical(),
                'expected_time': proj.time_of_flight_analytical(),
            })
    
    return pd.DataFrame(test_cases)