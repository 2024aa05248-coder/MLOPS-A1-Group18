# Part 4 — Model Packaging & Reproducibility

Scope
- Train, select, and package a final model with full preprocessing to ensure end-to-end reproducibility.
- Persist artifacts in reusable formats:
  - Joblib pipeline (preprocessing + estimator)
  - MLflow model directory (signature, input example, pip requirements)
  - Schema/metadata and selection metrics
- Provide clean, pinned environment files for deterministic installs.
- Provide an inference script that consumes raw features identical to Part 1 output without target.

Artifacts Produced
- Part4/models/final_model.joblib         # scikit-learn Pipeline (ColumnTransformer + estimator)
- Part4/models/mlflow_model/              # MLflow pyfunc model directory
- Part4/models/schema.json                # transformed feature names + feature groups metadata
- Part4/metrics/final_report.json         # metrics, selection details, artifact paths
- Part4/outputs/                          # optional predictions written by infer.py
- Part4/requirements.txt                  # pinned runtime deps for packaging/inference
- Part4/environment.yml                   # Conda environment with pinned versions
- Part4/src/infer.py                      # CLI for batch inference on CSV
- Part4/src/package_model.py              # trains, selects, and writes above artifacts

Prerequisites
- Part 1 generated the interim dataset:
  - Part1/data/interim/heart_clean.csv

How to Package (from repo root)
1) Ensure Part 1 artifacts exist:
   python3 Part1/scripts/download_data.py
   python3 Part1/src/data_preprocess.py

2) Run packaging:
    - python3 Part4/src/package_model.py

What Packaging Does
- Trains Logistic Regression and Random Forest using the same Part 2 preprocessing (scaling + one-hot).
- Performs 5-fold Stratified CV and computes:
  - accuracy, precision, recall, ROC-AUC (mean and std)
  - OOF ROC-AUC (from cross_val_predict)
- Selects the final model by highest ROC-AUC mean (break ties via OOF AUC).
- Fits the final pipeline on all data and saves:
  - Joblib pipeline: Part4/models/final_model.joblib
  - MLflow model dir: Part4/models/mlflow_model
  - schema.json with transformed feature names and groups
  - final_report.json capturing metrics and selection details

Reproducibility:
- The saved joblib pipeline contains:
  - ColumnTransformer: 
    - StandardScaler on continuous numeric: age, trestbps, chol, thalach, oldpeak, ca
    - passthrough on binary numeric flags: sex, fbs, exang
    - OneHotEncoder on integer-coded categoricals: cp, restecg, slope, thal
  - Estimator: chosen between Logistic Regression and Random Forest
- schema.json documents ColumnTransformer output feature names and original groups.

Inference
- Prepare a CSV of raw features (no target). For a quick sample from Part 1:
  python3 - <<'PY'
import os, pandas as pd
os.makedirs('Part4/inputs', exist_ok=True)
df = pd.read_csv('Part1/data/interim/heart_clean.csv')
df.drop(columns=['target']).head(20).to_csv('Part4/inputs/sample_X.csv', index=False)
print('Wrote Part4/inputs/sample_X.csv')
PY

- Run inference (joblib model by default):
  python3 Part4/src/infer.py --input Part4/inputs/sample_X.csv
  # Output written to Part4/outputs/predictions.csv (columns: original features + prediction)

- Use MLflow model directory (pyfunc):
  python3 Part4/src/infer.py --input Part4/inputs/sample_X.csv --mlflow-model-dir Part4/models/mlflow_model