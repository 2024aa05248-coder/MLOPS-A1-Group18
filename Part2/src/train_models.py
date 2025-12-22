#!/usr/bin/env python3
"""
Part 2 — Train Logistic Regression and Random Forest on the interim dataset.

- Uses the same preprocessing done in Part2 features.py
- Tunes models with GridSearchCV (scoring=roc_auc) using 5-fold Stratified CV
- Outputs:
  * metrics JSON per model
  * OOF ROC curve plot per model
  * best joblib models
  * a summary CSV with mean CV metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from .features import load_interim_dataset, split_X_y, build_model_pipeline
except ImportError:
    from features import load_interim_dataset, split_X_y, build_model_pipeline

# Get Paths to output folders
def get_paths() -> Dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    outputs = root / "Part2" / "outputs"
    paths = {
        "metrics": outputs / "metrics",
        "plots": outputs / "plots",
        "models": outputs / "models",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# Model grids and helpers
def grid_logreg() -> Dict[str, List[Any]]:
    return {
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l2"],
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__max_iter": [1000],
        "clf__class_weight": [None, "balanced"],
    }


def grid_rf() -> Dict[str, List[Any]]:
    return {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__class_weight": [None, "balanced"],
        "clf__random_state": [42],
    }


GRID_MAP = {
    "logreg": grid_logreg(),
    "rf": grid_rf(),
}


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


#Grid Search
def run_grid_search(name: str, base_estimator, X, y) -> GridSearchCV:
    pipe = build_model_pipeline(base_estimator)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=GRID_MAP[name],
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=True,
    )
    gs.fit(X, y)
    return gs

#Cross Validation evaluation
def evaluate_cv(pipeline, X, y) -> Dict[str, Any]:
    scoring = SCORING
    cvres = cross_validate(
        pipeline,
        X,
        y,
        cv=CV,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "cv_splits": CV.get_n_splits(),
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


#Plot roc of various algos
def plot_roc_oof(pipeline, X, y, out_path: Path, title: str) -> float:
    y_proba = cross_val_predict(pipeline, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.0))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return float(roc_auc)

#Compute Confusion Matrix
def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 3.8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

#Save json
def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# Main
def train_and_report(name: str, base_estimator, X, y, paths: Dict[str, Path]) -> Dict[str, Any]:
    gs = run_grid_search(name, base_estimator, X, y)
    best_pipe = gs.best_estimator_

    cv_metrics = evaluate_cv(best_pipe, X, y)
    roc_png = paths["plots"] / f"{name}_roc_curve.png"
    oof_auc = plot_roc_oof(best_pipe, X, y, roc_png, f"{name.upper()} — OOF ROC")

    # Confusion matrix from out-of-fold predictions
    y_pred = cross_val_predict(best_pipe, X, y, cv=CV, method="predict", n_jobs=-1)
    cm_png = paths["plots"] / f"{name}_confusion_matrix.png"
    save_confusion_matrix(y, y_pred, cm_png)

    # Fit on all data and save model
    best_pipe.fit(X, y)
    model_path = paths["models"] / f"{name}_best.joblib"
    joblib.dump(best_pipe, model_path)

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
            "model_path": str(model_path.resolve()),
        },
    }
    save_json(paths["metrics"] / f"{name}_cv_metrics.json", report)
    return report


def main() -> int:
    paths = get_paths()

    df = load_interim_dataset()
    X, y = split_X_y(df)

    # Logistic Regression
    logreg_report = train_and_report("logreg", LogisticRegression(), X, y, paths)

    # Random Forest
    rf_report = train_and_report("rf", RandomForestClassifier(), X, y, paths)

    # Summary CSV
    rows = []
    for name, rep in [("logreg", logreg_report), ("rf", rf_report)]:
        m = rep["cv_metrics"]
        rows.append(
            {
                "model": name,
                "accuracy_mean": m["accuracy_mean"],
                "accuracy_std": m["accuracy_std"],
                "precision_mean": m["precision_mean"],
                "precision_std": m["precision_std"],
                "recall_mean": m["recall_mean"],
                "recall_std": m["recall_std"],
                "roc_auc_mean": m["roc_auc_mean"],
                "roc_auc_std": m["roc_auc_std"],
                "oof_roc_auc": rep["oof_roc_auc"],
            }
        )
    pd.DataFrame(rows).to_csv(paths["metrics"] / "scores_summary.csv", index=False)

    print("Part2 training and evaluation complete.")
    print(f"Metrics JSON: {paths['metrics'].resolve()}")
    print(f"Plots:        {paths['plots'].resolve()}")
    print(f"Models:       {paths['models'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
