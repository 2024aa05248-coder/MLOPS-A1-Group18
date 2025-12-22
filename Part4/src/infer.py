#!/usr/bin/env python3
"""
Part 4 — Inference utility

Loads a packaged final model pipeline (joblib or MLflow pyfunc) and performs inference
on a CSV of raw features (same schema as Part1 interim without the 'target' column).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import joblib

try:
    import mlflow.pyfunc  
except Exception:
    mlflow = None


def _project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    return df


def load_joblib_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Joblib model not found at: {path}")
    return joblib.load(path)


def load_mlflow_model(path: Path):
    if mlflow is None:
        raise RuntimeError("mlflow is not installed; install or use joblib model.")
    # Load a generic pyfunc model
    return mlflow.pyfunc.load_model(str(path))


def run_inference(model, df: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Returns:
      - predictions (labels)
      - probabilities for positive class if available, otherwise None
    """
    # Common .predict path
    preds = pd.Series(model.predict(df), name="prediction")

    # Probability if available
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba_vals = model.predict_proba(df)[:, 1]
            proba = pd.Series(proba_vals, name="proba")
    except Exception:
        # Some wrappers (like pyfunc) won't expose predict_proba directly
        proba = None

    return preds, proba


def main() -> int:
    root = _project_root_from_script()

    parser = argparse.ArgumentParser(description="Inference with packaged final model.")
    parser.add_argument("--input", type=Path, required=True, help="CSV containing raw features.")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "Part4" / "models" / "final_model.joblib",
        help="Path to joblib model pipeline.",
    )
    parser.add_argument(
        "--mlflow-model-dir",
        type=Path,
        default=None,
        help="If provided, load this MLflow pyfunc model directory instead of joblib.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Part4" / "outputs" / "predictions.csv",
        help="Where to save predictions CSV.",
    )
    args = parser.parse_args()

    # Prepare I/O
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = load_input_csv(args.input)

    # Load model
    if args.mlflow_model_dir:
        model = load_mlflow_model(args.mlflow_model_dir)
    else:
        model = load_joblib_model(args.model)

    # Predict
    preds, proba = run_inference(model, df)

    # Assemble output
    out = df.copy()
    out["prediction"] = preds
    if proba is not None:
        out["proba"] = proba

    out.to_csv(args.output, index=False)
    print(f"Wrote predictions to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
