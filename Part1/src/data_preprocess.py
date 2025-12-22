#!/usr/bin/env python3
"""
Data cleaning and preprocessing for the UCI Heart Disease (Cleveland) dataset.

Inputs:
- data/raw/processed.cleveland.data

Outputs:
- data/interim/heart_clean.csv        (typed + imputed + not encoded)
- data/processed/heart_encoded.csv    (one-hot encoded features + target)
- data/processed/feature_names.json   (list of encoded feature names)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Columns (UCI Cleveland)
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num"
]

# Categorical to one-hot later
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]

# Numeric (includes binary flags kept numeric)
NUMERIC_COLS = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak",
    "ca",
]


def project_root() -> Path:
    # src/data_preprocess.py -> project root is parent of 'src'
    return Path(__file__).resolve().parents[1]


def load_raw(inpath: Path) -> pd.DataFrame:
    print(f"Loading dataset: {inpath.resolve()}")
    if not inpath.exists():
        raise FileNotFoundError(f"Input file not found: {inpath}")
    df = pd.read_csv(inpath, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)
    print(f"Raw shape: {df.shape}")
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    if "num" not in df.columns:
        raise KeyError("'num' column missing in dataset")
    return (df["num"] > 0).astype(int)


def impute(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    - Split features/target
    - Impute numeric (median) and categorical (most_frequent)
    - Keep numeric as numeric and categorical as one-hot encode ready
    """
    y = make_target(df)
    X = df.drop(columns=["num"]).copy()

    # Sanity check
    missing = set(NUMERIC_COLS + CATEGORICAL_COLS) - set(X.columns)
    if missing:
        raise KeyError(f"Expected columns missing: {missing}")

    # Numeric (fill missing value with median of the column)
    num_imputer = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(
        num_imputer.fit_transform(X[NUMERIC_COLS]),
        columns=NUMERIC_COLS,
        index=X.index,
    )
    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    # Categorical (fill missing value with most frequent value of that column)
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_cat = pd.DataFrame(
        cat_imputer.fit_transform(X[CATEGORICAL_COLS]),
        columns=CATEGORICAL_COLS,
        index=X.index,
    )

    X_clean = pd.concat([X_num, X_cat], axis=1)
    print(f"Imputed features shape: {X_clean.shape}")
    return X_clean, y


def encode_features(X_clean: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode categorical columns and return numeric feature DataFrame + names.
    """
    # Numeric as array
    X_num = X_clean[NUMERIC_COLS].to_numpy()

    # One-hot categorical
    # turn each categorical column into one column per unique value
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_cat = X_clean[CATEGORICAL_COLS]
    X_cat_enc = encoder.fit_transform(X_cat)

    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERIC_COLS + cat_feature_names

    # Combine Numeric and Encoded Categorical Features
    X_all = np.concatenate([X_num, X_cat_enc], axis=1)
    X_df = pd.DataFrame(X_all, columns=feature_names, index=X_clean.index)
    
    print(f"Encoded features shape: {X_df.shape}")
    return X_df, feature_names


def main() -> int:

    #Setup argument parser
    parser = argparse.ArgumentParser(description="Clean, impute, and encode UCI Heart Disease dataset.")
    default_root = project_root()
    parser.add_argument("--inpath", type=Path, default=default_root / "data" / "raw" / "processed.cleveland.data")
    parser.add_argument("--interim_dir", type=Path, default=default_root / "data" / "interim")
    parser.add_argument("--processed_dir", type=Path, default=default_root / "data" / "processed")
    args = parser.parse_args()

    # Ensure dirs exist
    args.interim_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw dataset
    df_raw = load_raw(args.inpath)

    #  Impute Missing Values & Split Target
    X_clean, y = impute(df_raw)

    # Save interim dataset (typed + imputed + target) (not yet encoded)
    interim_out = args.interim_dir / "heart_clean.csv"
    df_interim = X_clean.copy()
    df_interim["target"] = y.values
    df_interim.to_csv(interim_out, index=False)
    print(f"Saved interim dataset: {interim_out.resolve()}")

    # Encode features
    X_enc, feature_names = encode_features(X_clean)

    # Save final processed data (encoded + target)
    processed_out = args.processed_dir / "heart_encoded.csv"
    df_processed = X_enc.copy()
    df_processed["target"] = y.values
    df_processed.to_csv(processed_out, index=False)
    print(f"Saved processed dataset: {processed_out.resolve()}")

    # Save feature names
    feature_names_out = args.processed_dir / "feature_names.json"
    feature_names_out.write_text(json.dumps(feature_names, indent=2))
    print(f"Saved feature names to: {feature_names_out.resolve()}")

    print("Preprocessing done. Next: run 'python3 src/eda.py'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
