"""
Prediction utilities for projectile motion ML models.
Includes comparison between physics and ML predictions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os

from .model import BaseModel, create_model, PhysicsInformedModel


@dataclass
class PredictionResult:
    """Result from a single prediction."""
    ml_prediction: float
    physics_prediction: Optional[float] = None
    actual_value: Optional[float] = None
    ml_error: Optional[float] = None
    physics_error: Optional[float] = None
    
    @property
    def ml_is_better(self) -> Optional[bool]:
        """Check if ML prediction is closer to actual than physics."""
        if self.ml_error is not None and self.physics_error is not None:
            return abs(self.ml_error) < abs(self.physics_error)
        return None


class Predictor:
    """
    Unified predictor for projectile motion.
    Supports both ML models and physics-based predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model file
        """
        self.model: Optional[BaseModel] = None
        self.feature_names: List[str] = ['v0', 'angle', 'mass', 'drag_coef', 'radius']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model from file."""
        # Detect model type from extension or try to load
        self.model = create_model('linear')  # Create dummy to use load method
        self.model.load(model_path)
        self.feature_names = self.model.feature_names
        print(f"Loaded model: {self.model.name}")
    
    def predict_ml(self, v0: float, angle: float, mass: float = 1.0,
                   drag_coef: float = 0.47, radius: float = 0.05) -> float:
        """
        Make prediction using ML model.
        
        Args:
            v0: Initial velocity (m/s)
            angle: Launch angle (degrees)
            mass: Projectile mass (kg)
            drag_coef: Drag coefficient
            radius: Projectile radius (m)
            
        Returns:
            Predicted value (depends on what model was trained for)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        X = np.array([[v0, angle, mass, drag_coef, radius]])
        
        # Add physics features if model expects them
        if len(self.model.feature_names) > 5:
            X, _ = PhysicsInformedModel.add_physics_features(X, self.feature_names)
        
        return self.model.predict(X)[0]
    
    def predict_physics_no_drag(self, v0: float, angle: float) -> Dict[str, float]:
        """
        Calculate physics predictions without drag (analytical).
        """
        g = 9.81
        angle_rad = np.radians(angle)
        
        range_val = (v0**2 * np.sin(2 * angle_rad)) / g
        max_height = (v0 * np.sin(angle_rad))**2 / (2 * g)
        time_flight = 2 * v0 * np.sin(angle_rad) / g
        
        return {
            'range': range_val,
            'max_height': max_height,
            'time_of_flight': time_flight,
        }
    
    def predict_physics_with_drag(self, v0: float, angle: float, mass: float,
                                   drag_coef: float, radius: float,
                                   dt: float = 0.001) -> Dict[str, float]:
        """
        Calculate physics predictions with quadratic drag (numerical).
        """
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from physics.air_drag import ProjectileWithDrag, DragModel
        
        proj = ProjectileWithDrag(
            v0=v0, angle=angle, mass=mass,
            drag_coefficient=drag_coef, radius=radius
        )
        
        trajectory = proj.trajectory_rk4(dt=dt, drag_model=DragModel.QUADRATIC)
        
        if trajectory:
            return {
                'range': trajectory[-1].x,
                'max_height': max(p.y for p in trajectory),
                'time_of_flight': trajectory[-1].t,
                'final_speed': trajectory[-1].speed,
            }
        return {'range': 0, 'max_height': 0, 'time_of_flight': 0, 'final_speed': 0}
    
    def compare_predictions(self, v0: float, angle: float, mass: float = 1.0,
                            drag_coef: float = 0.47, radius: float = 0.05,
                            actual_value: Optional[float] = None) -> Dict:
        """
        Compare ML and physics predictions.
        
        Returns dict with both predictions and comparison metrics.
        """
        # Physics predictions
        physics_no_drag = self.predict_physics_no_drag(v0, angle)
        physics_with_drag = self.predict_physics_with_drag(v0, angle, mass, drag_coef, radius)
        
        result = {
            'inputs': {
                'v0': v0, 'angle': angle, 'mass': mass,
                'drag_coef': drag_coef, 'radius': radius
            },
            'physics_no_drag': physics_no_drag,
            'physics_with_drag': physics_with_drag,
            'drag_effect': {
                'range_reduction_pct': 100 * (1 - physics_with_drag['range'] / physics_no_drag['range']),
                'height_reduction_pct': 100 * (1 - physics_with_drag['max_height'] / physics_no_drag['max_height']),
            }
        }
        
        # ML prediction if model is loaded
        if self.model is not None:
            ml_pred = self.predict_ml(v0, angle, mass, drag_coef, radius)
            result['ml_prediction'] = ml_pred
            
            if actual_value is not None:
                result['actual'] = actual_value
                result['ml_error'] = ml_pred - actual_value
                result['ml_error_pct'] = 100 * (ml_pred - actual_value) / actual_value
        
        return result


def batch_predict(model: BaseModel, df: pd.DataFrame,
                  feature_columns: List[str] = None) -> np.ndarray:
    """
    Make predictions for a batch of inputs.
    
    Args:
        model: Trained model
        df: DataFrame with input features
        feature_columns: Columns to use as features
        
    Returns:
        Array of predictions
    """
    if feature_columns is None:
        feature_columns = ['v0', 'angle', 'mass', 'drag_coef', 'radius']
    
    X = df[feature_columns].values
    
    # Add physics features if needed
    if len(model.feature_names) > len(feature_columns):
        X, _ = PhysicsInformedModel.add_physics_features(X, feature_columns)
    
    return model.predict(X)


def evaluate_against_physics(model: BaseModel, df: pd.DataFrame,
                              target_col: str = 'range_quad_drag') -> pd.DataFrame:
    """
    Evaluate ML model against physics simulation results.
    
    Returns DataFrame with predictions and error analysis.
    """
    feature_cols = ['v0', 'angle', 'mass', 'drag_coef', 'radius']
    
    # Make predictions
    predictions = batch_predict(model, df, feature_cols)
    
    # Compare
    results_df = df[feature_cols + [target_col]].copy()
    results_df['ml_prediction'] = predictions
    results_df['physics_actual'] = df[target_col]
    results_df['error'] = predictions - df[target_col]
    results_df['error_pct'] = 100 * results_df['error'] / df[target_col]
    results_df['abs_error'] = np.abs(results_df['error'])
    
    # Add no-drag analytical for reference
    if 'range_no_drag' in df.columns:
        results_df['no_drag_range'] = df['range_no_drag']
        results_df['drag_effect'] = df['range_no_drag'] - df[target_col]
    
    return results_df


def get_prediction_summary(results_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics from prediction results.
    """
    return {
        'mean_error': results_df['error'].mean(),
        'std_error': results_df['error'].std(),
        'mean_abs_error': results_df['abs_error'].mean(),
        'max_abs_error': results_df['abs_error'].max(),
        'mean_error_pct': results_df['error_pct'].mean(),
        'correlation': results_df['ml_prediction'].corr(results_df['physics_actual']),
        'samples': len(results_df),
    }