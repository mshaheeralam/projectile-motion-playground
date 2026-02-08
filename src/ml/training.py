"""
Training pipeline for projectile motion ML models.
Supports both scikit-learn and PyTorch models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
import json
import time

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

from .model import (
    BaseModel, create_model, compare_models,
    LinearRegressionModel, MLPModel, RandomForestModel,
    PhysicsInformedModel, ModelMetrics
)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # Data split
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # Features and targets
    feature_columns: List[str] = field(default_factory=lambda: ['v0', 'angle', 'mass', 'drag_coef', 'radius'])
    target_column: str = 'range_quad_drag'
    
    # Model settings
    model_type: str = 'mlp'
    model_params: Dict = field(default_factory=dict)
    
    # Training
    use_physics_features: bool = False
    scale_features: bool = True
    
    # Output
    save_model: bool = True
    model_dir: str = 'models'


@dataclass
class TrainingResult:
    """Results from training."""
    model: BaseModel
    train_metrics: ModelMetrics
    val_metrics: Optional[ModelMetrics]
    test_metrics: ModelMetrics
    training_time_s: float
    config: TrainingConfig
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model.name,
            'train': self.train_metrics.to_dict(),
            'val': self.val_metrics.to_dict() if self.val_metrics else None,
            'test': self.test_metrics.to_dict(),
            'training_time_s': self.training_time_s,
        }


class TrainingPipeline:
    """
    End-to-end training pipeline for projectile motion prediction.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.model: Optional[BaseModel] = None
        self.scaler = StandardScaler()
        self.results: Optional[TrainingResult] = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features and target from dataframe.
        
        Returns:
            X: Features array
            y: Target array
            feature_names: List of feature names
        """
        cfg = self.config
        
        # Ensure all columns exist
        missing = [c for c in cfg.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")
        
        if cfg.target_column not in df.columns:
            raise ValueError(f"Target column '{cfg.target_column}' not in dataframe")
        
        X = df[cfg.feature_columns].values
        y = df[cfg.target_column].values
        
        # Add physics features if requested
        feature_names = cfg.feature_columns.copy()
        if cfg.use_physics_features:
            X, feature_names = PhysicsInformedModel.add_physics_features(X, feature_names)
        
        return X, y, feature_names
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data into train/val/test sets."""
        cfg = self.config
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state
        )
        
        # Second split: train vs val
        if cfg.val_size > 0:
            val_ratio = cfg.val_size / (1 - cfg.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_ratio, random_state=cfg.random_state
            )
        else:
            X_train, y_train = X_trainval, y_trainval
            X_val, y_val = None, None
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
        }
    
    def train(self, df: pd.DataFrame) -> TrainingResult:
        """
        Full training pipeline.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            TrainingResult with trained model and metrics
        """
        cfg = self.config
        
        # Prepare data
        X, y, feature_names = self.prepare_data(df)
        splits = self.split_data(X, y)
        
        # Create model
        self.model = create_model(cfg.model_type, **cfg.model_params)
        
        # Train
        start_time = time.time()
        self.model.fit(
            splits['X_train'], splits['y_train'],
            feature_names=feature_names,
            scale=cfg.scale_features
        )
        training_time = time.time() - start_time
        
        # Evaluate
        train_metrics = self.model.evaluate(
            splits['X_train'], splits['y_train'], scale=cfg.scale_features
        )
        
        val_metrics = None
        if splits['X_val'] is not None:
            val_metrics = self.model.evaluate(
                splits['X_val'], splits['y_val'], scale=cfg.scale_features
            )
        
        test_metrics = self.model.evaluate(
            splits['X_test'], splits['y_test'], scale=cfg.scale_features
        )
        
        # Save model
        if cfg.save_model:
            os.makedirs(cfg.model_dir, exist_ok=True)
            model_path = os.path.join(cfg.model_dir, f'{cfg.model_type}_{cfg.target_column}.pkl')
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        
        self.results = TrainingResult(
            model=self.model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            training_time_s=training_time,
            config=cfg
        )
        
        return self.results
    
    def cross_validate(self, df: pd.DataFrame, n_folds: int = 5) -> Dict:
        """
        Perform k-fold cross validation.
        """
        cfg = self.config
        X, y, feature_names = self.prepare_data(df)
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.random_state)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = create_model(cfg.model_type, **cfg.model_params)
            model.fit(X_train, y_train, feature_names=feature_names, scale=cfg.scale_features)
            
            metrics = model.evaluate(X_val, y_val, scale=cfg.scale_features)
            fold_results.append(metrics.to_dict())
            
            print(f"Fold {fold + 1}: RMSE = {metrics.rmse:.4f}, R² = {metrics.r2:.4f}")
        
        # Aggregate results
        return {
            'folds': fold_results,
            'mean_rmse': np.mean([r['rmse'] for r in fold_results]),
            'std_rmse': np.std([r['rmse'] for r in fold_results]),
            'mean_r2': np.mean([r['r2'] for r in fold_results]),
            'std_r2': np.std([r['r2'] for r in fold_results]),
        }


def train_all_targets(df: pd.DataFrame, 
                      targets: List[str] = None,
                      model_type: str = 'mlp') -> Dict[str, TrainingResult]:
    """
    Train models for multiple target variables.
    """
    if targets is None:
        targets = ['range_quad_drag', 'max_height_quad_drag', 'time_of_flight_quad_drag']
    
    results = {}
    
    for target in targets:
        if target not in df.columns:
            print(f"Warning: {target} not in dataframe, skipping")
            continue
            
        print(f"\n{'='*50}")
        print(f"Training for target: {target}")
        print('='*50)
        
        config = TrainingConfig(
            model_type=model_type,
            target_column=target,
            use_physics_features=True
        )
        
        pipeline = TrainingPipeline(config)
        results[target] = pipeline.train(df)
        
        print(f"Test RMSE: {results[target].test_metrics.rmse:.4f}")
        print(f"Test R²: {results[target].test_metrics.r2:.4f}")
    
    return results


def quick_train(df: pd.DataFrame, target: str = 'range_quad_drag',
                model_type: str = 'rf') -> Tuple[BaseModel, ModelMetrics]:
    """
    Quick training function for simple use cases.
    """
    config = TrainingConfig(
        model_type=model_type,
        target_column=target,
        save_model=False
    )
    
    pipeline = TrainingPipeline(config)
    result = pipeline.train(df)
    
    return result.model, result.test_metrics