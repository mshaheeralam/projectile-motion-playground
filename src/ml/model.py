"""
Machine Learning models for projectile motion prediction.
Includes baseline models (Linear Regression, MLP) and neural network models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import os

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""
    mse: float
    rmse: float
    mae: float
    r2: float
    
    def to_dict(self) -> Dict:
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2
        }


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None,
            target_names: List[str] = None,
            scale: bool = True) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Features array (n_samples, n_features)
            y: Target array (n_samples,) or (n_samples, n_targets)
            feature_names: Names of input features
            target_names: Names of target variables
            scale: Whether to standardize features
        """
        self.model = self._create_model()
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self.target_names = target_names or ['target']
        else:
            self.target_names = target_names or [f'target_{i}' for i in range(y.shape[1])]
        
        if scale:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)
        else:
            X_scaled, y_scaled = X, y
        
        self.model.fit(X_scaled, y_scaled.ravel() if y_scaled.shape[1] == 1 else y_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if scale:
            X_scaled = self.scaler_X.transform(X)
        else:
            X_scaled = X
        
        y_pred_scaled = self.model.predict(X_scaled)
        
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        if scale:
            return self.scaler_y.inverse_transform(y_pred_scaled).ravel()
        return y_pred_scaled.ravel()
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                 scale: bool = True) -> ModelMetrics:
        """Evaluate model performance."""
        y_pred = self.predict(X, scale)
        
        if y_true.ndim > 1:
            y_true = y_true.ravel()
        
        mse = mean_squared_error(y_true, y_pred)
        return ModelMetrics(
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mean_absolute_error(y_true, y_pred),
            r2=r2_score(y_true, y_pred)
        )
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'name': self.name,
            }, f)
    
    def load(self, filepath: str) -> 'BaseModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler_X = data['scaler_X']
        self.scaler_y = data['scaler_y']
        self.feature_names = data['feature_names']
        self.target_names = data['target_names']
        self.name = data['name']
        self.is_fitted = True
        return self


class LinearRegressionModel(BaseModel):
    """Simple linear regression baseline."""
    
    def __init__(self):
        super().__init__("Linear Regression")
    
    def _create_model(self):
        return LinearRegression()


class RidgeRegressionModel(BaseModel):
    """Ridge regression with L2 regularization."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("Ridge Regression")
        self.alpha = alpha
    
    def _create_model(self):
        return Ridge(alpha=self.alpha)


class RandomForestModel(BaseModel):
    """Random Forest regressor."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def _create_model(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return dict(zip(self.feature_names, self.model.feature_importances_))


class MLPModel(BaseModel):
    """Multi-Layer Perceptron (sklearn version)."""
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (64, 32),
                 max_iter: int = 1000, learning_rate_init: float = 0.001):
        super().__init__("MLP Regressor")
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
    
    def _create_model(self):
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )


class GradientBoostingModel(BaseModel):
    """Gradient Boosting regressor."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 3,
                 learning_rate: float = 0.1):
        super().__init__("Gradient Boosting")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def _create_model(self):
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )


class PhysicsInformedModel(BaseModel):
    """
    Physics-informed model that uses analytical solutions as features.
    
    The idea: use physics to compute no-drag solution analytically,
    then let ML learn the correction factor for drag effects.
    
    This is a hybrid approach that combines domain knowledge with ML.
    """
    
    def __init__(self, base_model: str = 'mlp'):
        super().__init__("Physics-Informed Model")
        self.base_model_type = base_model
    
    def _create_model(self):
        if self.base_model_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=(32, 16),
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
        elif self.base_model_type == 'rf':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            return LinearRegression()
    
    @staticmethod
    def add_physics_features(X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Add physics-derived features based on analytical solutions.
        
        Assumes X contains: v0, angle, mass, drag_coef, radius
        """
        # Find feature indices
        idx = {name: i for i, name in enumerate(feature_names)}
        
        v0 = X[:, idx['v0']]
        angle_rad = np.radians(X[:, idx['angle']])
        mass = X[:, idx.get('mass', 0)] if 'mass' in idx else np.ones(len(X))
        drag_coef = X[:, idx.get('drag_coef', 0)] if 'drag_coef' in idx else np.zeros(len(X))
        radius = X[:, idx.get('radius', 0)] if 'radius' in idx else 0.05 * np.ones(len(X))
        
        g = 9.81
        
        # Analytical solutions (no drag)
        range_analytical = (v0**2 * np.sin(2 * angle_rad)) / g
        max_height_analytical = (v0 * np.sin(angle_rad))**2 / (2 * g)
        time_analytical = 2 * v0 * np.sin(angle_rad) / g
        
        # Physics-based drag indicators
        area = np.pi * radius**2
        drag_factor = 0.5 * 1.225 * drag_coef * area / mass  # ρ * Cd * A / (2m)
        reynolds_proxy = v0 * radius  # Proxy for Reynolds number
        
        # Combine original features with physics features
        physics_features = np.column_stack([
            X,
            range_analytical,
            max_height_analytical,
            time_analytical,
            drag_factor,
            reynolds_proxy,
            v0 * np.cos(angle_rad),  # vx0
            v0 * np.sin(angle_rad),  # vy0
        ])
        
        new_names = feature_names + [
            'range_analytical', 'height_analytical', 'time_analytical',
            'drag_factor', 'reynolds_proxy', 'vx0', 'vy0'
        ]
        
        return physics_features, new_names


def create_model(model_type: str, **kwargs) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'linear', 'ridge', 'rf', 'mlp', 'gb', 'physics'
        **kwargs: Additional arguments for the model
    """
    models = {
        'linear': LinearRegressionModel,
        'ridge': RidgeRegressionModel,
        'rf': RandomForestModel,
        'mlp': MLPModel,
        'gb': GradientBoostingModel,
        'physics': PhysicsInformedModel,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   feature_names: List[str] = None,
                   models: List[str] = None) -> pd.DataFrame:
    """
    Train and compare multiple models.
    
    Returns DataFrame with metrics for each model.
    """
    if models is None:
        models = ['linear', 'ridge', 'rf', 'mlp', 'gb']
    
    results = []
    
    for model_type in models:
        print(f"Training {model_type}...")
        model = create_model(model_type)
        model.fit(X_train, y_train, feature_names=feature_names)
        
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        results.append({
            'model': model.name,
            'train_rmse': train_metrics.rmse,
            'test_rmse': test_metrics.rmse,
            'train_r2': train_metrics.r2,
            'test_r2': test_metrics.r2,
            'train_mae': train_metrics.mae,
            'test_mae': test_metrics.mae,
        })
    
    return pd.DataFrame(results)