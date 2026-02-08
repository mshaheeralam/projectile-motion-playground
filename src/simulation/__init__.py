"""
Simulation module for projectile motion.
"""
from .engine import (
    SimulationEngine, SimulationConfig, SimulationResult,
    SolutionMethod, quick_simulation
)
from .data_generator import (
    DatasetGenerator, DatasetConfig,
    generate_training_data, generate_test_cases
)

__all__ = [
    'SimulationEngine', 'SimulationConfig', 'SimulationResult',
    'SolutionMethod', 'quick_simulation',
    'DatasetGenerator', 'DatasetConfig',
    'generate_training_data', 'generate_test_cases',
]