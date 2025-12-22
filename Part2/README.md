# Part 2 — Feature Engineering & Model Development

Scope
- Prepare final ML features (scaling + encoding)
- Train and tune at least two classifiers (Logistic Regression, Random Forest)
- Evaluate with stratified cross-validation using accuracy, precision, recall, ROC-AUC
- Save metrics, plots, and best models

Prerequisites
- Complete Part 1 to generate the interim dataset:
  - Part1/data/interim/heart_clean.csv
- Install dependencies at the project root:
  - python3 -m pip install -r requirements.txt

Project Structure
- Part2/src/features.py
  - Loads Part1 interim data, defines feature groups
  - ColumnTransformer:
    - StandardScaler for continuous numeric
    - passthrough for binary numeric flags
    - OneHotEncoder for integer-coded categoricals
  - Helper to build a preprocessing+estimator Pipeline
- Part2/src/train_models.py
  - Loads data from Part1
  - GridSearchCV for Logistic Regression and Random Forest
  - Cross-validated metrics: accuracy, precision, recall, ROC-AUC
  - Out-of-fold ROC curves
  - Saves artifacts to Part2/outputs

How to Run
From repository root (mlops-heart-disease):

1) Ensure Part 1 data exists
- If you haven’t run Part 1 after the folder reorg, do:
  python3 Part1/scripts/download_data.py
  python3 Part1/src/data_preprocess.py

This will create:
- Part1/data/interim/heart_clean.csv

2) Train models and generate metrics/artifacts
- Run Part 2 training (as a module so relative imports resolve):
  python3 Part2/src/train_models.py

Outputs
- Part2/outputs/metrics/
  - logreg_cv_metrics.json
  - rf_cv_metrics.json
  - scores_summary.csv
- Part2/outputs/plots/
  - logreg_roc_curve.png
  - rf_roc_curve.png
- Part2/outputs/models/
  - logreg_best.joblib
  - rf_best.joblib

Notes:
- Feature groups (features.py):
  - Continuous numeric (scaled): age, trestbps, chol, thalach, oldpeak, ca
  - Binary numeric (passthrough): sex, fbs, exang
  - Categorical (one-hot): cp, restecg, slope, thal
- Model selection (train_models.py):
  - Logistic Regression (liblinear, L2): grid over C, class_weight
  - Random Forest: grid over n_estimators, max_depth, min_samples_leaf, class_weight
- Cross-validation:
  - StratifiedKFold with 5 splits, shuffle=True, random_state=42
  - Metrics computed via cross_validate; OOF ROC via cross_val_predict
