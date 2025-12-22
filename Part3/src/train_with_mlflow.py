#!/usr/bin/env python3
"""
Part 3 — Track experiments with MLflow using the Part2 preprocessing pipeline.

- Loads Part1 interim dataset (imputed + target)
- Reuses Part2 preprocessing (scale continuous, passthrough binary, one-hot categoricals)
- Tunes Logistic Regression and Random Forest with GridSearchCV (scoring=roc_auc, 5-fold Stratified CV)
- Logs to MLflow: params, metrics, ROC plot, JSON report, and the fitted pipeline model
- Saves local artifacts under Part3/outputs/

MLflow UI (from repo root):
  mlflow ui --backend-store-uri file://$(pwd)/Part3/mlruns
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import joblib

import mlflow
import mlflow.sklearn

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Part2.src.features import (
    load_interim_dataset,
    split_X_y,
    NUMERIC_CONT_COLS,
    BINARY_NUMERIC_COLS,
    CATEGORICAL_COLS,
)
from Part2.src.train_models import (
    CV,
    run_grid_search,
    evaluate_cv,
    plot_roc_oof,
    save_confusion_matrix,
)

# Ouptut Paths
def get_paths() -> Dict[str, Path]:
    outputs = PROJECT_ROOT / "Part3" / "outputs"
    paths = {
        "metrics": outputs / "metrics",
        "plots": outputs / "plots",
        "models": outputs / "models",
        "mlruns": PROJECT_ROOT / "Part3" / "mlruns",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

# Param Logging
def log_common_params() -> None:
    mlflow.log_param("num_cont_cols", ",".join(NUMERIC_CONT_COLS))
    mlflow.log_param("binary_numeric_cols", ",".join(BINARY_NUMERIC_COLS))
    mlflow.log_param("categorical_cols", ",".join(CATEGORICAL_COLS))

# Model Run logging
def log_model_run(name: str, base_estimator, X, y, paths: Dict[str, Path]) -> Dict[str, Any]:
    gs = run_grid_search(name, base_estimator, X, y)
    best_pipe = gs.best_estimator_

    cv_metrics = evaluate_cv(best_pipe, X, y)
    roc_png = paths["plots"] / f"{name}_roc_curve.png"
    oof_auc = plot_roc_oof(best_pipe, X, y, roc_png, f"{name.upper()} — OOF ROC")
    # Confusion matrix from out-of-fold predictions
    y_pred = cross_val_predict(best_pipe, X, y, cv=CV, method="predict", n_jobs=-1)
    cm_png = paths["plots"] / f"{name}_confusion_matrix.png"
    save_confusion_matrix(y, y_pred, cm_png)

    best_pipe.fit(X, y)
    model_path = paths["models"] / f"{name}_best.joblib"
    joblib.dump(best_pipe, model_path)

    mlflow.log_param("model_name", name)
    mlflow.log_params(gs.best_params_)
    mlflow.log_metric("best_cv_score", float(gs.best_score_))
    mlflow.log_metrics(
        {
            "accuracy_mean": cv_metrics["accuracy_mean"],
            "precision_mean": cv_metrics["precision_mean"],
            "recall_mean": cv_metrics["recall_mean"],
            "roc_auc_mean": cv_metrics["roc_auc_mean"],
            "oof_roc_auc": oof_auc,
        }
    )

    mlflow.log_artifact(str(roc_png))
    mlflow.log_artifact(str(cm_png))

    report = {
        "model": "logistic_regression" if name == "logreg" else "random_forest",
        "selection": {
            "best_score": float(gs.best_score_),
            "best_params": gs.best_params_,
            "cv_splits": CV.get_n_splits(),
            "scoring": "roc_auc",
        },
        "cv_metrics": cv_metrics,
        "oof_roc_auc": oof_auc,
        "artifacts": {
            "roc_plot": str(roc_png.resolve()),
            "confusion_matrix": str(cm_png.resolve()),
            "local_model_path": str(model_path.resolve()),
        },
    }
    report_path = paths["metrics"] / f"{name}_mlflow_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    mlflow.log_artifact(str(report_path))

    mlflow.sklearn.log_model(sk_model=best_pipe, artifact_path="model")
    return report


def main() -> int:
    paths = get_paths()

    # MLflow config
    mlflow.set_tracking_uri(f"file://{paths['mlruns'].resolve()}")
    mlflow.set_experiment("HeartDisease-MLflow")

    # Data
    df = load_interim_dataset()
    X, y = split_X_y(df)

    parent_params = {"cv_splits": 5, "scoring": "roc_auc", "random_state": 42, "notes": "Part3 MLflow tracking."}

    with mlflow.start_run(run_name="Part3 Training"):
        mlflow.log_params(parent_params)
        log_common_params()

        with mlflow.start_run(run_name="logreg", nested=True):
            log_common_params()
            log_model_run("logreg", LogisticRegression(), X, y, paths)

        with mlflow.start_run(run_name="rf", nested=True):
            log_common_params()
            log_model_run("rf", RandomForestClassifier(), X, y, paths)

    print("Part3 MLflow tracking complete.")
    print(f"MLflow tracking URI: file://{paths['mlruns'].resolve()}")
    print(f"To view UI: mlflow ui --backend-store-uri file://{paths['mlruns'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
