#!/usr/bin/env python3
"""
Unit tests for inference functionality.
Tests model loading, prediction pipeline, and output format.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class TestInferenceSetup:
    """Test inference setup and model loading."""
    
    def test_final_model_exists(self):
        """Test that final packaged model exists."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        assert model_path.exists(), "Final model not found in Part4/models/"
    
    def test_model_can_be_loaded(self):
        """Test that final model can be loaded."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            assert model is not None
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
    
    def test_schema_exists(self):
        """Test that schema file exists."""
        schema_path = PROJECT_ROOT / "Part4" / "models" / "schema.json"
        assert schema_path.exists(), "Schema file not found"


class TestInferencePipeline:
    """Test inference pipeline functionality."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input data for inference."""
        return pd.DataFrame({
            'age': [63, 67, 45],
            'sex': [1, 1, 0],
            'cp': [1, 4, 3],
            'trestbps': [145, 160, 130],
            'chol': [233, 286, 250],
            'fbs': [1, 0, 0],
            'restecg': [2, 2, 1],
            'thalach': [150, 108, 170],
            'exang': [0, 1, 0],
            'oldpeak': [2.3, 1.5, 1.0],
            'slope': [3, 2, 2],
            'ca': [0, 3, 1],
            'thal': [6, 3, 3],
        })
    
    def test_inference_on_sample_data(self, sample_input):
        """Test that inference works on sample data."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            predictions = model.predict(sample_input)
            
            assert len(predictions) == len(sample_input)
            assert all(p in [0, 1] for p in predictions)
    
    def test_inference_probabilities(self, sample_input):
        """Test that inference returns valid probabilities."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            probas = model.predict_proba(sample_input)
            
            assert probas.shape == (len(sample_input), 2)
            assert np.all((probas >= 0) & (probas <= 1))
            assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_single_sample_inference(self):
        """Test inference on a single sample."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            
            single_sample = pd.DataFrame({
                'age': [63],
                'sex': [1],
                'cp': [1],
                'trestbps': [145],
                'chol': [233],
                'fbs': [1],
                'restecg': [2],
                'thalach': [150],
                'exang': [0],
                'oldpeak': [2.3],
                'slope': [3],
                'ca': [0],
                'thal': [6],
            })
            
            prediction = model.predict(single_sample)
            assert len(prediction) == 1
            assert prediction[0] in [0, 1]


class TestInferenceValidation:
    """Test input validation for inference."""
    
    def test_missing_columns_detection(self):
        """Test that missing columns are detected."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            
            # Create input with missing column
            incomplete_input = pd.DataFrame({
                'age': [63],
                'sex': [1],
                'cp': [1],
                # Missing other columns
            })
            
            # Should raise an error
            with pytest.raises(Exception):
                model.predict(incomplete_input)
    
    def test_extra_columns_handling(self):
        """Test that extra columns don't break inference."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            
            # Create input with extra column
            input_with_extra = pd.DataFrame({
                'age': [63],
                'sex': [1],
                'cp': [1],
                'trestbps': [145],
                'chol': [233],
                'fbs': [1],
                'restecg': [2],
                'thalach': [150],
                'exang': [0],
                'oldpeak': [2.3],
                'slope': [3],
                'ca': [0],
                'thal': [6],
                'extra_column': [999],  # Extra column
            })
            
            # Should work (extra columns typically ignored)
            try:
                prediction = model.predict(input_with_extra)
                assert len(prediction) == 1
            except KeyError:
                # Some pipelines might be strict about columns
                pass


class TestInferenceOutput:
    """Test inference output format and quality."""
    
    def test_output_is_deterministic(self):
        """Test that predictions are deterministic."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            
            sample = pd.DataFrame({
                'age': [63],
                'sex': [1],
                'cp': [1],
                'trestbps': [145],
                'chol': [233],
                'fbs': [1],
                'restecg': [2],
                'thalach': [150],
                'exang': [0],
                'oldpeak': [2.3],
                'slope': [3],
                'ca': [0],
                'thal': [6],
            })
            
            pred1 = model.predict(sample)
            pred2 = model.predict(sample)
            
            assert np.array_equal(pred1, pred2)
    
    def test_batch_inference_consistency(self):
        """Test that batch inference is consistent with individual predictions."""
        model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            
            samples = pd.DataFrame({
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
            
            # Batch prediction
            batch_pred = model.predict(samples)
            
            # Individual predictions
            ind_pred1 = model.predict(samples.iloc[[0]])
            ind_pred2 = model.predict(samples.iloc[[1]])
            
            assert batch_pred[0] == ind_pred1[0]
            assert batch_pred[1] == ind_pred2[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

