#!/usr/bin/env python3
"""
Unit tests for model training and evaluation functionality.
Tests model training, prediction, metrics, and model persistence.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Part2.src.features import build_model_pipeline


class TestModelTraining:
    """Test model training functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame({
            'age': np.random.randint(30, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(1, 5, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(150, 400, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(70, 200, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6, n_samples),
            'slope': np.random.randint(1, 4, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(3, 8, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
    
    def test_logistic_regression_training(self, sample_data):
        """Test that Logistic Regression can be trained."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        pipe.fit(X, y)
        
        assert hasattr(pipe, 'predict')
        predictions = pipe.predict(X)
        assert len(predictions) == len(y)
    
    def test_random_forest_training(self, sample_data):
        """Test that Random Forest can be trained."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        pipe = build_model_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
        pipe.fit(X, y)
        
        assert hasattr(pipe, 'predict')
        predictions = pipe.predict(X)
        assert len(predictions) == len(y)
    
    def test_model_predict_proba(self, sample_data):
        """Test that models can output probabilities."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        pipe.fit(X, y)
        
        probas = pipe.predict_proba(X)
        assert probas.shape == (len(X), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_cross_validation(self, sample_data):
        """Test that cross-validation works."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        scores = cross_val_score(pipe, X, y, cv=3, scoring='accuracy')
        
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)


class TestModelEvaluation:
    """Test model evaluation metrics."""
    
    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        acc = accuracy_score(y_true, y_pred)
        assert 0 <= acc <= 1
        assert acc == 0.8  # 4 out of 5 correct
    
    def test_precision_calculation(self):
        """Test precision metric calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        prec = precision_score(y_true, y_pred)
        assert 0 <= prec <= 1
    
    def test_recall_calculation(self):
        """Test recall metric calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        rec = recall_score(y_true, y_pred)
        assert 0 <= rec <= 1
    
    def test_roc_auc_calculation(self):
        """Test ROC-AUC metric calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.85])
        
        auc = roc_auc_score(y_true, y_scores)
        assert 0 <= auc <= 1


class TestModelPersistence:
    """Test model saving and loading."""
    
    @pytest.fixture
    def trained_model(self, tmp_path):
        """Create and train a simple model."""
        X = pd.DataFrame({
            'age': [63, 67, 67, 37, 41],
            'sex': [1, 1, 1, 1, 0],
            'cp': [1, 4, 4, 3, 2],
            'trestbps': [145, 160, 120, 130, 130],
            'chol': [233, 286, 229, 250, 204],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [2, 2, 2, 0, 2],
            'thalach': [150, 108, 129, 187, 172],
            'exang': [0, 1, 1, 0, 0],
            'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
            'slope': [3, 2, 2, 3, 1],
            'ca': [0, 3, 2, 0, 0],
            'thal': [6, 3, 7, 3, 3],
        })
        y = pd.Series([0, 1, 1, 0, 0])
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        pipe.fit(X, y)
        
        return pipe, X
    
    def test_model_save_load(self, trained_model, tmp_path):
        """Test that model can be saved and loaded."""
        pipe, X = trained_model
        model_path = tmp_path / "test_model.joblib"
        
        # Save model
        joblib.dump(pipe, model_path)
        assert model_path.exists()
        
        # Load model
        loaded_pipe = joblib.load(model_path)
        assert loaded_pipe is not None
        
        # Test predictions match
        orig_pred = pipe.predict(X)
        loaded_pred = loaded_pipe.predict(X)
        assert np.array_equal(orig_pred, loaded_pred)
    
    def test_saved_models_exist(self):
        """Test that trained models were saved in Part2."""
        models_dir = PROJECT_ROOT / "Part2" / "outputs" / "models"
        if models_dir.exists():
            logreg_path = models_dir / "logreg_best.joblib"
            rf_path = models_dir / "rf_best.joblib"
            
            # At least one model should exist
            assert logreg_path.exists() or rf_path.exists()


class TestModelPredictions:
    """Test model prediction functionality."""
    
    def test_prediction_output_format(self):
        """Test that predictions are in correct format."""
        X = pd.DataFrame({
            'age': [63, 67],
            'sex': [1, 1],
            'cp': [1, 4],
            'trestbps': [145, 160],
            'chol': [233, 286],
            'fbs': [1, 0],
            'restecg': [2, 2],
            'thalach': [150, 108],
            'exang': [0, 1],
            'oldpeak': [2.3, 1.5],
            'slope': [3, 2],
            'ca': [0, 3],
            'thal': [6, 3],
        })
        y = pd.Series([0, 1])
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        pipe.fit(X, y)
        
        predictions = pipe.predict(X)
        
        # Check output format
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)
    
    def test_probability_output_format(self):
        """Test that probability predictions are in correct format."""
        X = pd.DataFrame({
            'age': [63, 67],
            'sex': [1, 1],
            'cp': [1, 4],
            'trestbps': [145, 160],
            'chol': [233, 286],
            'fbs': [1, 0],
            'restecg': [2, 2],
            'thalach': [150, 108],
            'exang': [0, 1],
            'oldpeak': [2.3, 1.5],
            'slope': [3, 2],
            'ca': [0, 3],
            'thal': [6, 3],
        })
        y = pd.Series([0, 1])
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        pipe.fit(X, y)
        
        probas = pipe.predict_proba(X)
        
        # Check output format
        assert probas.shape == (len(X), 2)
        assert np.all((probas >= 0) & (probas <= 1))
        assert np.allclose(probas.sum(axis=1), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

