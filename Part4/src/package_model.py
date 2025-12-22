#!/usr/bin/env python3
"""
Part 4 — Model Packaging & Reproducibility

Goals:
- Train and select a final model (LogReg vs RandomForest) using the Part2 feature pipeline
- Persist the final model in reusable formats:
  * Joblib (sklearn pipeline with preprocessing + estimator)
  * MLflow model directory (pip requirements + signature + input example)
- Save reproducibility metadata (feature names, feature groups, best params, metrics)

Outputs:
- Part4/models/final_model.joblib
- Part4/models/mlflow_model/       (MLflow pyfunc model dir)
- Part4/models/schema.json          (feature names + groups)
- Part4/metrics/final_report.json   (selection, metrics, artifacts)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Part2.src.features import (
    load_interim_dataset,
    split_X_y,
    build_model_pipeline,
    get_feature_names_after_fit,
    NUMERIC_CONT_COLS,
    BINARY_NUMERIC_COLS,
    CATEGORICAL_COLS,
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


@dataclass
class Paths:
    root: Path
    models_dir: Path
    mlflow_model_dir: Path
    metrics_dir: Path
    plots_dir: Path
    schema_json: Path
    final_model_joblib: Path
    final_report_json: Path


def ensure_paths() -> Paths:
    root = PROJECT_ROOT
    models_dir = root / "Part4" / "models"
    mlflow_model_dir = models_dir / "mlflow_model"
    metrics_dir = root / "Part4" / "metrics"
    plots_dir = root / "Part4" / "plots"
    for p in (models_dir, mlflow_model_dir, metrics_dir, plots_dir):
        p.mkdir(parents=True, exist_ok=True)
    schema_json = models_dir / "schema.json"
    final_model_joblib = models_dir / "final_model.joblib"
    final_report_json = metrics_dir / "final_report.json"
    return Paths(
        root=root,
        models_dir=models_dir,
        mlflow_model_dir=mlflow_model_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
        schema_json=schema_json,
        final_model_joblib=final_model_joblib,
        final_report_json=final_report_json,
    )

# Parameter grid function for logistic regression
def param_grid_logreg() -> Dict[str, List[Any]]:
    return {
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l2"],
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__max_iter": [1000],
        "clf__class_weight": [None, "balanced"],
    }

# Parameter grid function for random forest
def param_grid_rf() -> Dict[str, List[Any]]:
    return {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__class_weight": [None, "balanced"],
        "clf__random_state": [42],
    }


def grid_search(
    name: str,
    base_estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    verbose: int = 0,
) -> Tuple[GridSearchCV, Dict[str, Any]]:
    pipeline = build_model_pipeline(base_estimator)
    grid = param_grid_logreg() if name == "logreg" else param_grid_rf()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
        return_train_score=True,
    )
    gs.fit(X, y)
    selection = {
        "best_score": float(gs.best_score_),
        "best_params": gs.best_params_,
        "cv_splits": cv_splits,
        "scoring": scoring,
        "model_name": name,
    }
    return gs, selection


def evaluate_cv(pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> Dict[str, Any]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    cvres = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "cv_splits": cv_splits,
        "accuracy_mean": float(np.mean(cvres["test_accuracy"])),
        "accuracy_std": float(np.std(cvres["test_accuracy"])),
        "precision_mean": float(np.mean(cvres["test_precision"])),
        "precision_std": float(np.std(cvres["test_precision"])),
        "recall_mean": float(np.mean(cvres["test_recall"])),
        "recall_std": float(np.std(cvres["test_recall"])),
        "roc_auc_mean": float(np.mean(cvres["test_roc_auc"])),
        "roc_auc_std": float(np.std(cvres["test_roc_auc"])),
        "per_fold": {
            "accuracy": cvres["test_accuracy"].tolist(),
            "precision": cvres["test_precision"].tolist(),
            "recall": cvres["test_recall"].tolist(),
            "roc_auc": cvres["test_roc_auc"].tolist(),
        },
    }


def plot_roc_oof(pipeline, X: pd.DataFrame, y: pd.Series, out_path: Path, title: str) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.0))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return float(roc_auc)


def save_schema(paths: Paths, feature_names: List[str]) -> None:
    schema = {
        "feature_names": feature_names,
        "groups": {
            "numeric_cont": NUMERIC_CONT_COLS,
            "binary_numeric": BINARY_NUMERIC_COLS,
            "categorical": CATEGORICAL_COLS,
        },
        "notes": "feature_names correspond to ColumnTransformer output order after preprocessing.",
    }
    paths.schema_json.write_text(json.dumps(schema, indent=2))

# Saves the pipeline in MLflow format
def save_mlflow_model(
    pipeline, X_sample: pd.DataFrame, paths: Paths, model_name: str, metrics: Dict[str, Any], selection: Dict[str, Any]
) -> None:
    # Create a minimal input example (first 5 rows of raw features)
    input_example = X_sample.iloc[:5].copy()
    # Model signature (input: raw features, output: predicted probabilities or class - we use predict_proba proba[:,1])
    # For a pyfunc model with sklearn pipeline classifier, default predict returns class labels.
    signature = infer_signature(
        input_example,
        pd.Series([0.0], name="proba", dtype=float),
    )

    # Minimal pip requirements to reload the model
    pip_requirements = [
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "mlflow>=2.12.1",
        "joblib>=1.3.0",
    ]

    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=str(paths.mlflow_model_dir),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        metadata={
            "model_name": model_name,
            "selection": selection,
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()},
        },
    )


def main() -> int:
    paths = ensure_paths()

    # Load data
    df = load_interim_dataset()
    X, y = split_X_y(df)

    # Optimization: Reuse best trained model from Part2 if available (avoid retraining)
    try:
        part2_root = PROJECT_ROOT / "Part2" / "outputs"
        p2_metrics_dir = part2_root / "metrics"
        p2_models_dir = part2_root / "models"
        summary_csv = p2_metrics_dir / "scores_summary.csv"

        if summary_csv.exists() and p2_models_dir.exists():
            # Decide best model by roc_auc_mean (tie-breaker: oof_roc_auc)
            _summary = pd.read_csv(summary_csv)
            sort_cols = [c for c in ["roc_auc_mean", "oof_roc_auc"] if c in _summary.columns]
            if sort_cols:
                _summary = _summary.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            best_name = str(_summary.iloc[0]["model"]).strip()

            model_file = "logreg_best.joblib" if best_name == "logreg" else "rf_best.joblib"
            model_path = p2_models_dir / model_file

            if model_path.exists():
                # Load trained pipeline (includes preprocessing)
                best_pipe = joblib.load(model_path)

                # Save final joblib under Part4
                joblib.dump(best_pipe, paths.final_model_joblib)

                # Save schema (transformed feature names) and MLflow model dir
                feat_names = get_feature_names_after_fit(best_pipe)
                save_schema(paths, feat_names)

                # Load metrics from Part2 JSON for report
                metrics_json_path = p2_metrics_dir / ("logreg_cv_metrics.json" if best_name == "logreg" else "rf_cv_metrics.json")
                best_cv = {}
                best_oof = 0.0
                best_sel = {}
                if metrics_json_path.exists():
                    _best_report = json.loads(metrics_json_path.read_text())
                    best_cv = _best_report.get("cv_metrics", {}) or {}
                    best_oof = float(_best_report.get("oof_roc_auc", 0.0) or 0.0)
                    best_sel = _best_report.get("selection", {}) or {}

                save_mlflow_model(
                    pipeline=best_pipe,
                    X_sample=X,
                    paths=paths,
                    model_name=f"final_{best_name}",
                    metrics={"roc_auc_mean": float(best_cv.get("roc_auc_mean", 0.0)), "oof_roc_auc": best_oof},
                    selection=best_sel,
                )

                # Optionally include both models' metrics in the final report if present
                def _load_json(p: Path):
                    try:
                        return json.loads(p.read_text()) if p.exists() else None
                    except Exception:
                        return None

                logreg_json = _load_json(p2_metrics_dir / "logreg_cv_metrics.json")
                rf_json = _load_json(p2_metrics_dir / "rf_cv_metrics.json")

                final_report = {
                    "chosen_model": best_name,
                    "artifacts": {
                        "final_model_joblib": str(paths.final_model_joblib.resolve()),
                        "mlflow_model_dir": str(paths.mlflow_model_dir.resolve()),
                        "schema_json": str(paths.schema_json.resolve()),
                        "plots_dir": str(paths.plots_dir.resolve()),
                    },
                    "metrics": {
                        **({"logreg": {"cv": logreg_json.get("cv_metrics"), "oof_roc_auc": logreg_json.get("oof_roc_auc"), "selection": logreg_json.get("selection")}} if logreg_json else {}),
                        **({"rf": {"cv": rf_json.get("cv_metrics"), "oof_roc_auc": rf_json.get("oof_roc_auc"), "selection": rf_json.get("selection")}} if rf_json else {}),
                        "final": {"cv": best_cv, "oof_roc_auc": best_oof, "selection": best_sel},
                    },
                    "reproducibility_notes": {
                        "random_state": 42,
                        "cv_splits": 5,
                        "preprocessing": {
                            "numeric_cont_scaled": NUMERIC_CONT_COLS,
                            "binary_numeric_passthrough": BINARY_NUMERIC_COLS,
                            "categorical_onehot": CATEGORICAL_COLS,
                        },
                        "source": "Reused trained model from Part2",
                    },
                }
                paths.final_report_json.write_text(json.dumps(final_report, indent=2))

                print("[INFO] Part4 packaging complete (reused Part2 trained model).")
                print(f"[INFO] Final joblib pipeline: {paths.final_model_joblib.resolve()}")
                print(f"[INFO] MLflow model dir:      {paths.mlflow_model_dir.resolve()}")
                print(f"[INFO] Schema JSON:           {paths.schema_json.resolve()}")
                print(f"[INFO] Final report JSON:     {paths.final_report_json.resolve()}")
                return 0
    except Exception as _reuse_err:
        print(f"[WARN] Reuse of Part2 model failed, falling back to retraining in Part4: {_reuse_err}")

    # Train and select best of two models
    logreg_gs, sel_logreg = grid_search("logreg", LogisticRegression(), X, y, cv_splits=5, scoring="roc_auc")
    rf_gs, sel_rf = grid_search("rf", RandomForestClassifier(), X, y, cv_splits=5, scoring="roc_auc")

    # Evaluate both (with CV metrics and OOF ROC)
    logreg_cv = evaluate_cv(logreg_gs.best_estimator_, X, y, cv_splits=5)
    rf_cv = evaluate_cv(rf_gs.best_estimator_, X, y, cv_splits=5)

    logreg_roc = plot_roc_oof(logreg_gs.best_estimator_, X, y, paths.plots_dir / "logreg_roc_curve.png", "LogReg — OOF ROC")
    rf_roc = plot_roc_oof(rf_gs.best_estimator_, X, y, paths.plots_dir / "rf_roc_curve.png", "RF — OOF ROC")

    # Pick best by roc_auc_mean (ties -> choose higher oof_roc)
    def score_tuple(cv: Dict[str, Any], oof_auc: float) -> Tuple[float, float]:
        return (cv["roc_auc_mean"], oof_auc)

    best_name = "logreg"
    best_gs = logreg_gs
    best_cv = logreg_cv
    best_sel = sel_logreg
    best_oof = logreg_roc

    if score_tuple(rf_cv, rf_roc) > score_tuple(logreg_cv, logreg_roc):
        best_name = "rf"
        best_gs = rf_gs
        best_cv = rf_cv
        best_sel = sel_rf
        best_oof = rf_roc

    best_pipe = best_gs.best_estimator_
    best_pipe.fit(X, y)

    # Persist joblib pipeline (preprocessing + estimator end-to-end)
    joblib.dump(best_pipe, paths.final_model_joblib)

    # Save schema metadata (feature names after fit)
    feat_names = get_feature_names_after_fit(best_pipe)
    save_schema(paths, feat_names)

    # Save MLflow model directory with signature + requirements
    save_mlflow_model(
        pipeline=best_pipe,
        X_sample=X,
        paths=paths,
        model_name=f"final_{best_name}",
        metrics={"roc_auc_mean": best_cv["roc_auc_mean"], "oof_roc_auc": best_oof},
        selection=best_sel,
    )

    # Write final report JSON
    final_report = {
        "chosen_model": best_name,
        "artifacts": {
            "final_model_joblib": str(paths.final_model_joblib.resolve()),
            "mlflow_model_dir": str(paths.mlflow_model_dir.resolve()),
            "schema_json": str(paths.schema_json.resolve()),
            "plots_dir": str(paths.plots_dir.resolve()),
        },
        "metrics": {
            "logreg": {"cv": logreg_cv, "oof_roc_auc": logreg_roc, "selection": sel_logreg},
            "rf": {"cv": rf_cv, "oof_roc_auc": rf_roc, "selection": sel_rf},
            "final": {"cv": best_cv, "oof_roc_auc": best_oof, "selection": best_sel},
        },
        "reproducibility_notes": {
            "random_state": 42,
            "cv_splits": 5,
            "preprocessing": {
                "numeric_cont_scaled": NUMERIC_CONT_COLS,
                "binary_numeric_passthrough": BINARY_NUMERIC_COLS,
                "categorical_onehot": CATEGORICAL_COLS,
            },
        },
    }
    paths.final_report_json.write_text(json.dumps(final_report, indent=2))

    print("[INFO] Part4 packaging complete.")
    print(f"[INFO] Final joblib pipeline: {paths.final_model_joblib.resolve()}")
    print(f"[INFO] MLflow model dir:      {paths.mlflow_model_dir.resolve()}")
    print(f"[INFO] Schema JSON:           {paths.schema_json.resolve()}")
    print(f"[INFO] Final report JSON:     {paths.final_report_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
