#!/usr/bin/env python3
"""
Part 2 feature utilities.

- Preprocessor: scale continuous, passthrough binary flags, one-hot encode categoricals
- Helpers to load interim dataset and split X/y
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Columns as defined/used in Part1
TARGET_COL: str = "target"

# Treat as continuous numeric (to be scaled)
NUMERIC_CONT_COLS: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# Binary numeric flags (keep as numeric but do not scale)
BINARY_NUMERIC_COLS: List[str] = ["sex", "fbs", "exang"]

# Integer-coded categoricals (will be one-hot encoded)
CATEGORICAL_COLS: List[str] = ["cp", "restecg", "slope", "thal"]


# Project root (parent of 'Part2')
ROOT: Path = Path(__file__).resolve().parents[2]


def default_interim_path() -> Path:
    # Default to Part1 interim clean dataset
    return ROOT / "Part1" / "data" / "interim" / "heart_clean.csv"


def load_interim_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the interim dataset (typed + imputed + target) produced by Part1."""
    inpath = path or default_interim_path()
    if not inpath.exists():
        raise FileNotFoundError(f"Interim dataset not found at: {inpath}")
    df = pd.read_csv(inpath)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}' missing in dataset.")
    return df


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and target y."""
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - scales continuous numeric columns
      - leaves binary numeric flags as-is
      - one-hot encodes categorical columns (robust to unseen categories)
    """
    numeric_cont_transformer = StandardScaler()

    # For binary numeric flags, passthrough keeps them unchanged
    binary_numeric_transformer = "passthrough"

    # One-hot for categorical integer-coded features
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_cont", numeric_cont_transformer, NUMERIC_CONT_COLS),
            ("bin_num", binary_numeric_transformer, BINARY_NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def build_model_pipeline(estimator) -> Pipeline:
    """
    Attach a classifier to the preprocessing pipeline.
    """
    pre = build_preprocessor()
    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("clf", estimator),
        ]
    )
    return pipe


def get_feature_names_after_fit(pipeline: Pipeline) -> List[str]:
    """
    Return the transformed feature names after fitting the pipeline.
    Must be called AFTER pipeline.fit(...).
    """
    preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
    try:
        names = preprocess.get_feature_names_out().tolist()
        return names
    except Exception:
        # Fallback: construct names manually
        names: List[str] = []
        # num_cont
        names.extend([f"num_cont__{c}" for c in NUMERIC_CONT_COLS])
        # bin_num
        names.extend([f"bin_num__{c}" for c in BINARY_NUMERIC_COLS])
        # cat: approximate without categories
        names.extend([f"cat__{c}" for c in CATEGORICAL_COLS])
        return names


if __name__ == "__main__":
    # Small self-check when run directly
    df = load_interim_dataset()
    X, y = split_X_y(df)
    from sklearn.linear_model import LogisticRegression

    pipe = build_model_pipeline(LogisticRegression(max_iter=200))
    pipe.fit(X, y)
    feat_names = get_feature_names_after_fit(pipe)
    print(f"Fitted pipeline. Transformed feature count: {len(feat_names)}")
