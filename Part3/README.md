# Part 3 — Experiment Tracking with MLflow

Scope
- Integrate MLflow for experiment tracking (parameters, metrics, artifacts, models).
- Reuse Part 2 preprocessing and training utilities (no code duplication).
- Train and track two models:
  - Logistic Regression
  - Random Forest

Code Flow and outputs
- Part3/src/train_with_mlflow.py now imports training/evaluation helpers directly from Part2:
  - from Part2.src.train_models import CV, run_grid_search, evaluate_cv, plot_roc_oof, save_confusion_matrix
- Part 3 focuses only on:
  - MLflow setup (tracking URI, experiment)
  - Running model training via shared helpers
  - Logging params/metrics/artifacts/model to MLflow
  - Writing Part 3 artifacts under Part3/outputs

Prerequisites
- Complete Part 1 to generate the interim dataset:
  - Part1/data/interim/heart_clean.csv
- Install dependencies at repo root:
  - python3 -m pip install -r requirements.txt

Project Structure
- Part3/src/train_with_mlflow.py
  - Loads Part1 interim data
  - Reuses Part2/src/features.py preprocessing (columns logged as params)
  - Uses Part2/src/train_models.py utilities for CV, grid search, ROC/CM plotting, metrics
  - Logs with MLflow:
    - Params: best hyperparams, CV setup, feature groupings
    - Metrics: CV means (accuracy, precision, recall, roc_auc), OOF ROC-AUC
    - Artifacts: ROC and confusion matrix plots, JSON report
    - Model: logged via mlflow.sklearn.log_model
  - MLflow local file store: Part3/mlruns

How to Run
From the repository root:

1) Ensure Part 1 artifacts exist
- If not already done:
  python Part1/scripts/download_data.py
  python Part1/src/data_preprocess.py

This creates:
- Part1/data/interim/heart_clean.csv

2) Run Part 3 with MLflow tracking
- As a module (recommended):
  python Part3/src/train_with_mlflow

MLflow UI
- Start the UI from repo root:
  mlflow ui --backend-store-uri file://$(pwd)/Part3/mlruns
- Open http://127.0.0.1:5000

Outputs
- Part3/mlruns/                # MLflow tracking data (experiments/runs)
- Part3/outputs/metrics/
  - logreg_mlflow_report.json
  - rf_mlflow_report.json
- Part3/outputs/plots/
  - logreg_roc_curve.png
  - rf_roc_curve.png
- Part3/outputs/models/
  - logreg_best.joblib
  - rf_best.joblib

