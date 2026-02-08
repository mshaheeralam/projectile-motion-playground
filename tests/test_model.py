"""
Tests for machine learning models.
"""
import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.model import (
    LinearRegressionModel, RidgeRegressionModel,
    RandomForestModel, MLPModel, ModelMetrics
)
from simulation.data_generator import DatasetGenerator, DatasetConfig
from physics.air_drag import DragModel


class TestModelMetrics:
    """Test ModelMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ModelMetrics(mse=1.0, rmse=1.0, mae=0.8, r2=0.95)
        d = metrics.to_dict()
        
        assert d['mse'] == 1.0
        assert d['rmse'] == 1.0
        assert d['mae'] == 0.8
        assert d['r2'] == 0.95


class TestLinearRegressionModel:
    """Test LinearRegressionModel."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple linear data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
        return X, y
    
    def test_fit_predict(self, simple_data):
        """Test basic fit and predict."""
        X, y = simple_data
        model = LinearRegressionModel()
        
        model.fit(X[:80], y[:80])
        predictions = model.predict(X[80:])
        
        assert len(predictions) == 20
        assert model.is_fitted
    
    def test_evaluate_returns_metrics(self, simple_data):
        """Test evaluation returns proper metrics."""
        X, y = simple_data
        model = LinearRegressionModel()
        
        model.fit(X[:80], y[:80])
        metrics = model.evaluate(X[80:], y[80:])
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.r2 > 0.9  # Should fit well on linear data
    
    def test_save_and_load(self, simple_data):
        """Test model persistence."""
        X, y = simple_data
        model = LinearRegressionModel()
        model.fit(X[:80], y[:80], feature_names=['a', 'b', 'c'])
        
        predictions_before = model.predict(X[80:])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model.pkl')
            model.save(filepath)
            
            loaded_model = LinearRegressionModel()
            loaded_model.load(filepath)
            
            predictions_after = loaded_model.predict(X[80:])
        
        assert np.allclose(predictions_before, predictions_after)
        assert loaded_model.feature_names == ['a', 'b', 'c']


class TestRidgeRegressionModel:
    """Test RidgeRegressionModel."""
    
    def test_regularization_effect(self):
        """Ridge should produce smaller coefficients than OLS."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = X[:, 0] + np.random.randn(50) * 0.1
        
        linear = LinearRegressionModel()
        ridge = RidgeRegressionModel(alpha=10.0)
        
        linear.fit(X, y, scale=False)
        ridge.fit(X, y, scale=False)
        
        # Ridge should produce more regularized (smaller) coefficients
        linear_coef_norm = np.linalg.norm(linear.model.coef_)
        ridge_coef_norm = np.linalg.norm(ridge.model.coef_)
        
        assert ridge_coef_norm < linear_coef_norm


class TestRandomForestModel:
    """Test RandomForestModel."""
    
    @pytest.fixture
    def nonlinear_data(self):
        """Generate nonlinear data."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X[:, 0]**2 + np.sin(X[:, 1]) + X[:, 2] + np.random.randn(200) * 0.1
        return X, y
    
    def test_fit_predict(self, nonlinear_data):
        """Test basic fit and predict."""
        X, y = nonlinear_data
        model = RandomForestModel(n_estimators=10)
        
        model.fit(X[:160], y[:160])
        predictions = model.predict(X[160:])
        
        assert len(predictions) == 40
    
    def test_feature_importance(self, nonlinear_data):
        """Test feature importance extraction."""
        X, y = nonlinear_data
        model = RandomForestModel(n_estimators=50)
        model.fit(X, y, feature_names=['x1', 'x2', 'x3'])
        
        importance = model.feature_importance()
        
        assert len(importance) == 3
        assert 'x1' in importance
        assert sum(importance.values()) > 0
    
    def test_handles_nonlinear_patterns(self, nonlinear_data):
        """RF should handle nonlinear patterns better than linear."""
        X, y = nonlinear_data
        
        linear = LinearRegressionModel()
        rf = RandomForestModel(n_estimators=50)
        
        linear.fit(X[:160], y[:160])
        rf.fit(X[:160], y[:160])
        
        linear_metrics = linear.evaluate(X[160:], y[160:])
        rf_metrics = rf.evaluate(X[160:], y[160:])
        
        # RF should perform better on nonlinear data
        assert rf_metrics.r2 > linear_metrics.r2


class TestMLPModel:
    """Test MLPModel."""
    
    @pytest.fixture
    def complex_data(self):
        """Generate more complex nonlinear data."""
        np.random.seed(42)
        X = np.random.randn(300, 4)
        y = (X[:, 0]**2 + X[:, 1]*X[:, 2] - X[:, 3] + 
             np.random.randn(300) * 0.2)
        return X, y
    
    def test_fit_predict(self, complex_data):
        """Test basic fit and predict."""
        X, y = complex_data
        model = MLPModel(hidden_layers=(32, 16), max_iter=500)
        
        model.fit(X[:240], y[:240])
        predictions = model.predict(X[240:])
        
        assert len(predictions) == 60
    
    def test_different_architectures(self, complex_data):
        """Test different hidden layer configurations."""
        X, y = complex_data
        
        model_small = MLPModel(hidden_layers=(16,), max_iter=500)
        model_large = MLPModel(hidden_layers=(64, 32, 16), max_iter=500)
        
        model_small.fit(X[:240], y[:240])
        model_large.fit(X[:240], y[:240])
        
        # Both should produce predictions
        assert len(model_small.predict(X[240:])) == 60
        assert len(model_large.predict(X[240:])) == 60


class TestModelsWithPhysicsData:
    """Test models on actual physics simulation data."""
    
    @pytest.fixture
    def physics_data(self):
        """Generate physics simulation data."""
        config = DatasetConfig(
            num_samples=500,
            include_no_drag=False,
            include_linear_drag=False,
            include_quadratic_drag=True,
            seed=42
        )
        generator = DatasetGenerator(config)
        df = generator.generate(show_progress=False)
        
        X = df[['v0', 'angle', 'mass']].values
        y = df['range_quad_drag'].values
        
        return X, y
    
    def test_linear_regression_on_physics_data(self, physics_data):
        """Linear regression baseline on physics data."""
        X, y = physics_data
        model = LinearRegressionModel()
        
        model.fit(X[:400], y[:400])
        metrics = model.evaluate(X[400:], y[400:])
        
        # Should achieve reasonable fit
        assert metrics.r2 > 0.5
    
    def test_rf_on_physics_data(self, physics_data):
        """Random forest on physics data."""
        X, y = physics_data
        model = RandomForestModel(n_estimators=50)
        
        model.fit(X[:400], y[:400])
        metrics = model.evaluate(X[400:], y[400:])
        
        # Should achieve good fit
        assert metrics.r2 > 0.8
    
    def test_mlp_on_physics_data(self, physics_data):
        """MLP on physics data."""
        X, y = physics_data
        model = MLPModel(hidden_layers=(64, 32), max_iter=1000)
        
        model.fit(X[:400], y[:400])
        metrics = model.evaluate(X[400:], y[400:])
        
        # Should achieve good fit
        assert metrics.r2 > 0.7
    
    def test_model_comparison(self, physics_data):
        """Compare all models on same data."""
        X, y = physics_data
        
        models = [
            ('Linear', LinearRegressionModel()),
            ('Ridge', RidgeRegressionModel(alpha=1.0)),
            ('RF', RandomForestModel(n_estimators=50)),
            ('MLP', MLPModel(hidden_layers=(32, 16), max_iter=500))
        ]
        
        results = {}
        for name, model in models:
            model.fit(X[:400], y[:400])
            metrics = model.evaluate(X[400:], y[400:])
            results[name] = metrics.r2
        
        # All models should have positive R²
        for name, r2 in results.items():
            assert r2 > 0, f"{name} should have positive R²"


class TestModelNotFittedError:
    """Test error handling for unfitted models."""
    
    def test_predict_without_fit_raises_error(self):
        """Predicting without fitting should raise error."""
        model = LinearRegressionModel()
        X = np.array([[1, 2, 3]])
        
        with pytest.raises(RuntimeError, match="Model not fitted"):
            model.predict(X)
    
    def test_evaluate_without_fit_raises_error(self):
        """Evaluating without fitting should raise error."""
        model = RandomForestModel()
        X = np.array([[1, 2, 3]])
        y = np.array([1.0])
        
        with pytest.raises(RuntimeError, match="Model not fitted"):
            model.evaluate(X, y)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])