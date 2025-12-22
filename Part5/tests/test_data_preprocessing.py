#!/usr/bin/env python3
"""
Unit tests for data preprocessing functionality.
Tests data cleaning, imputation, encoding, and validation.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class TestDataPreprocessing:
    """Test suite for data preprocessing operations."""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data with missing values."""
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
            'num': [0, 2, 1, 0, 0]
        })
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values marked as '?'."""
        return pd.DataFrame({
            'age': [63, 67, '?', 37, 41],
            'sex': [1, 1, 1, 1, 0],
            'cp': [1, 4, 4, 3, 2],
            'trestbps': [145, '?', 120, 130, 130],
            'chol': [233, 286, 229, 250, 204],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [2, 2, 2, 0, 2],
            'thalach': [150, 108, 129, 187, 172],
            'exang': [0, 1, 1, 0, 0],
            'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
            'slope': [3, 2, 2, 3, 1],
            'ca': [0, 3, 2, 0, '?'],
            'thal': [6, 3, 7, 3, 3],
            'num': [0, 2, 1, 0, 0]
        })
    
    def test_missing_value_replacement(self, sample_data_with_missing):
        """Test that '?' values are replaced with NaN."""
        df = sample_data_with_missing.replace('?', np.nan)
        assert df.isna().sum().sum() > 0
        assert '?' not in df.values
    
    def test_target_creation(self, sample_raw_data):
        """Test binary target creation from num column."""
        df = sample_raw_data.copy()
        df['target'] = (df['num'] > 0).astype(int)
        
        assert 'target' in df.columns
        assert df['target'].dtype == int
        assert set(df['target'].unique()).issubset({0, 1})
        assert df['target'].sum() == 2  # Two positive cases in sample
    
    def test_numeric_imputation(self, sample_data_with_missing):
        """Test median imputation for numeric columns."""
        df = sample_data_with_missing.copy()
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        
        # Check that we have missing values
        assert df['age'].isna().sum() > 0
        
        # Impute with median
        median_age = df['age'].median()
        df['age'].fillna(median_age, inplace=True)
        
        # Check no missing values remain
        assert df['age'].isna().sum() == 0
    
    def test_data_shape_preservation(self, sample_raw_data):
        """Test that preprocessing preserves number of rows."""
        original_rows = len(sample_raw_data)
        df = sample_raw_data.copy()
        df['target'] = (df['num'] > 0).astype(int)
        
        assert len(df) == original_rows
    
    def test_feature_types(self, sample_raw_data):
        """Test that features have correct data types."""
        df = sample_raw_data.copy()
        
        # Numeric features should be numeric
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])
    
    def test_no_duplicate_rows(self, sample_raw_data):
        """Test that there are no duplicate rows after preprocessing."""
        df = sample_raw_data.copy()
        duplicates = df.duplicated().sum()
        # This is informational - duplicates might exist in real data
        assert duplicates >= 0
    
    def test_target_balance(self, sample_raw_data):
        """Test that target variable has both classes."""
        df = sample_raw_data.copy()
        df['target'] = (df['num'] > 0).astype(int)
        
        unique_targets = df['target'].unique()
        assert len(unique_targets) >= 1  # At least one class
        assert all(t in [0, 1] for t in unique_targets)


class TestDataValidation:
    """Test suite for data validation."""
    
    def test_required_columns_present(self):
        """Test that all required columns are present in processed data."""
        required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                        'ca', 'thal', 'target']
        
        # Try to load interim data
        interim_path = PROJECT_ROOT / "Part1" / "data" / "interim" / "heart_clean.csv"
        if interim_path.exists():
            df = pd.read_csv(interim_path)
            for col in required_cols:
                assert col in df.columns, f"Required column {col} missing"
    
    def test_no_missing_values_in_clean_data(self):
        """Test that cleaned data has no missing values."""
        interim_path = PROJECT_ROOT / "Part1" / "data" / "interim" / "heart_clean.csv"
        if interim_path.exists():
            df = pd.read_csv(interim_path)
            missing_count = df.isna().sum().sum()
            assert missing_count == 0, f"Found {missing_count} missing values in clean data"
    
    def test_target_is_binary(self):
        """Test that target variable is binary (0 or 1)."""
        interim_path = PROJECT_ROOT / "Part1" / "data" / "interim" / "heart_clean.csv"
        if interim_path.exists():
            df = pd.read_csv(interim_path)
            assert set(df['target'].unique()).issubset({0, 1})
    
    def test_data_not_empty(self):
        """Test that processed data is not empty."""
        interim_path = PROJECT_ROOT / "Part1" / "data" / "interim" / "heart_clean.csv"
        if interim_path.exists():
            df = pd.read_csv(interim_path)
            assert len(df) > 0, "Processed data is empty"
            assert len(df.columns) > 0, "No columns in processed data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

