"""
Machine Learning module for projectile motion prediction.
"""
from .model import (
    BaseModel, ModelMetrics,
    LinearRegressionModel, RidgeRegressionModel,
    RandomForestModel, MLPModel, GradientBoostingModel,
    PhysicsInformedModel,
    create_model, compare_models
)
from .training import (
    TrainingConfig, TrainingResult, TrainingPipeline,
    train_all_targets, quick_train
)
from .prediction import (
    Predictor, PredictionResult,
    batch_predict, evaluate_against_physics, get_prediction_summary
)

__all__ = [
    # Models
    'BaseModel', 'ModelMetrics',
    'LinearRegressionModel', 'RidgeRegressionModel',
    'RandomForestModel', 'MLPModel', 'GradientBoostingModel',
    'PhysicsInformedModel',
    'create_model', 'compare_models',
    # Training
    'TrainingConfig', 'TrainingResult', 'TrainingPipeline',
    'train_all_targets', 'quick_train',
    # Prediction
    'Predictor', 'PredictionResult',
    'batch_predict', 'evaluate_against_physics', 'get_prediction_summary',
]