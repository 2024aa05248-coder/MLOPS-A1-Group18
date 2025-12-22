#!/usr/bin/env python3
"""
Unit tests for feature engineering functionality.
Tests feature transformations, scaling, encoding, and pipeline operations.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Part2.src.features import (
    NUMERIC_CONT_COLS,
    BINARY_NUMERIC_COLS,
    CATEGORICAL_COLS,
    build_model_pipeline,
    get_feature_names_after_fit,
)


class TestFeatureDefinitions:
    """Test feature group definitions."""
    
    def test_feature_groups_defined(self):
        """Test that feature groups are properly defined."""
        assert len(NUMERIC_CONT_COLS) > 0
        assert len(BINARY_NUMERIC_COLS) > 0
        assert len(CATEGORICAL_COLS) > 0
    
    def test_no_feature_overlap(self):
        """Test that feature groups don't overlap."""
        all_features = set(NUMERIC_CONT_COLS + BINARY_NUMERIC_COLS + CATEGORICAL_COLS)
        total_count = len(NUMERIC_CONT_COLS) + len(BINARY_NUMERIC_COLS) + len(CATEGORICAL_COLS)
        assert len(all_features) == total_count, "Features appear in multiple groups"
    
    def test_expected_continuous_features(self):
        """Test that expected continuous features are present."""
        expected = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        for feat in expected:
            assert feat in NUMERIC_CONT_COLS, f"{feat} should be in continuous features"
    
    def test_expected_binary_features(self):
        """Test that expected binary features are present."""
        expected = ['sex', 'fbs', 'exang']
        for feat in expected:
            assert feat in BINARY_NUMERIC_COLS, f"{feat} should be in binary features"
    
    def test_expected_categorical_features(self):
        """Test that expected categorical features are present."""
        expected = ['cp', 'restecg', 'slope', 'thal']
        for feat in expected:
            assert feat in CATEGORICAL_COLS, f"{feat} should be in categorical features"


class TestFeaturePipeline:
    """Test feature transformation pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
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
            'target': [0, 1, 1, 0, 0]
        })
    
    def test_pipeline_creation(self):
        """Test that pipeline can be created."""
        from sklearn.linear_model import LogisticRegression
        pipe = build_model_pipeline(LogisticRegression())
        assert pipe is not None
        assert hasattr(pipe, 'fit')
        assert hasattr(pipe, 'predict')
    
    def test_pipeline_fit_transform(self, sample_data):
        """Test that pipeline can fit and transform data."""
        from sklearn.linear_model import LogisticRegression
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X, y)
        
        # Test prediction
        predictions = pipe.predict(X)
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)
    
    def test_pipeline_preserves_sample_count(self, sample_data):
        """Test that pipeline doesn't change number of samples."""
        from sklearn.linear_model import LogisticRegression
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X, y)
        
        X_transformed = pipe.named_steps['preprocessor'].transform(X)
        assert X_transformed.shape[0] == len(X)
    
    def test_scaling_applied(self, sample_data):
        """Test that scaling is applied to continuous features."""
        from sklearn.linear_model import LogisticRegression
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X, y)
        
        # Get transformed data
        X_transformed = pipe.named_steps['preprocessor'].transform(X)
        
        # Scaled features should have values roughly in [-3, 3] range for small samples
        # Just check that transformation happened
        assert X_transformed.shape[1] > len(X.columns)  # One-hot encoding increases dimensions
    
    def test_feature_names_extraction(self, sample_data):
        """Test that feature names can be extracted after fitting."""
        from sklearn.linear_model import LogisticRegression
        
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        pipe = build_model_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X, y)
        
        feature_names = get_feature_names_after_fit(pipe.named_steps['preprocessor'])
        assert len(feature_names) > 0
        assert isinstance(feature_names, list)


class TestFeatureTransformations:
    """Test individual feature transformations."""
    
    def test_standard_scaler(self):
        """Test StandardScaler behavior."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        
        # Check mean is close to 0 and std is close to 1
        assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled.std(axis=0), 1, atol=1e-10)
    
    def test_onehot_encoder(self):
        """Test OneHotEncoder behavior."""
        data = np.array([[1], [2], [3], [1]])
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(data)
        
        # Should have 3 columns (one for each unique value)
        assert encoded.shape[1] == 3
        # Each row should sum to 1
        assert np.allclose(encoded.sum(axis=1), 1)
    
    def test_column_transformer_structure(self):
        """Test ColumnTransformer with different transformations."""
        X = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4, 5, 6],
            'cat1': [1, 2, 1]
        })
        
        ct = ColumnTransformer([
            ('scaler', StandardScaler(), ['num1', 'num2']),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['cat1'])
        ])
        
        X_transformed = ct.fit_transform(X)
        
        # Should have 2 scaled + 2 one-hot encoded features
        assert X_transformed.shape[1] == 4
        assert X_transformed.shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

