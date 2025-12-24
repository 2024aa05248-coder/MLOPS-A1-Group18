# Complete MLOps Assignment Setup Guide - Parts 1-8

This guide provides step-by-step instructions to complete the entire MLOps assignment from data preprocessing to deployment with monitoring.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Part 1: Data Preprocessing & EDA](#part-1-data-preprocessing--eda)
- [Part 2: Model Training](#part-2-model-training)
- [Part 3: MLflow Integration](#part-3-mlflow-integration)
- [Part 4: Model Packaging & Inference](#part-4-model-packaging--inference)
- [Part 5: Testing (Optional)](#part-5-testing-optional)
- [Part 6: API Development](#part-6-api-development)
- [Part 7 & 8: Deployment & Monitoring](#part-7--8-deployment--monitoring)

---

## Prerequisites

### Software Requirements
- **Python 3.9+**
- **Docker Desktop** (with Kubernetes enabled)
- **Minikube**
- **kubectl**
- **Git**

### Initial Setup

**Step 1: Clone/Navigate to Project Directory**
```powershell
cd "C:\Users\ashmitad\Documents\Personal\Bits\SEMESTER 3\MLOPs\Assignment 1\MLOP-Assign"
```

**Step 2: Create Virtual Environment**
```powershell
python -m venv venv
```

**Step 3: Activate Virtual Environment**
```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Step 4: Install Dependencies**
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify Installation:**
```powershell
python -c "import pandas, numpy, sklearn, mlflow, fastapi; print('All packages installed successfully!')"
```

---

## Part 1: Data Preprocessing & EDA

### Objective
Download, clean, and preprocess the UCI Heart Disease dataset.

### Step 1.1: Download Dataset

```powershell
cd Part1
python dataset_download_script\download_data.py
```

**Expected Output:**
- `data/raw/processed.cleveland.data` created

### Step 1.2: Data Preprocessing

```powershell
python src\data_preprocess.py
```

**Expected Output:**
```
✓ Loaded 303 rows from raw data
✓ Cleaned dataset: 297 rows
✓ Saved to data/interim/heart_clean.csv
✓ Encoded dataset: 297 rows, 23 features
✓ Saved to data/processed/heart_encoded.csv
✓ Feature names saved to data/processed/feature_names.json
```

**Verify Files Created:**
```powershell
ls data\interim\
ls data\processed\
```

### Step 1.3: Exploratory Data Analysis (EDA)

```powershell
python src\eda.py
```

**Expected Output:**
- `reports/figures/histograms_numeric.png`
- `reports/figures/corr_heatmap.png`
- `reports/figures/class_balance.png`

**View Generated Plots:**
```powershell
Start-Process reports\figures\
```

### Step 1.4: Verify Part 1 Completion

```powershell
# Check all required files exist
Test-Path data\interim\heart_clean.csv
Test-Path data\processed\heart_encoded.csv
Test-Path data\processed\feature_names.json
Test-Path reports\figures\histograms_numeric.png
```

All should return `True`.

---

## Part 2: Model Training

### Objective
Train Logistic Regression and Random Forest models with hyperparameter tuning.

### Step 2.1: Navigate to Part2

```powershell
cd ..\Part2
```

### Step 2.2: Train Models

```powershell
python src\train_models.py
```

**Expected Output:**
```
Training Logistic Regression...
  Best params: {'clf__C': 0.1, 'clf__penalty': 'l2', ...}
  Best CV ROC-AUC: 0.8XX
  OOF ROC-AUC: 0.8XX
  ✓ Saved model to outputs/models/logreg_best.joblib

Training Random Forest...
  Best params: {'clf__max_depth': 10, 'clf__n_estimators': 200, ...}
  Best CV ROC-AUC: 0.8XX
  OOF ROC-AUC: 0.8XX
  ✓ Saved model to outputs/models/rf_best.joblib

Summary saved to outputs/metrics/scores_summary.csv
```

**Training Time:** 2-5 minutes depending on your machine.

### Step 2.3: Verify Outputs

```powershell
# Check models
ls outputs\models\

# Check metrics
ls outputs\metrics\

# Check plots
ls outputs\plots\
Start-Process outputs\plots\
```

**Expected Files:**
- `outputs/models/logreg_best.joblib`
- `outputs/models/rf_best.joblib`
- `outputs/metrics/logreg_cv_metrics.json`
- `outputs/metrics/rf_cv_metrics.json`
- `outputs/metrics/scores_summary.csv`
- `outputs/plots/logreg_confusion_matrix.png`
- `outputs/plots/logreg_roc_curve.png`
- `outputs/plots/rf_confusion_matrix.png`
- `outputs/plots/rf_roc_curve.png`

### Step 2.4: Review Model Performance

```powershell
# View summary
Get-Content outputs\metrics\scores_summary.csv

# View detailed metrics
Get-Content outputs\metrics\rf_cv_metrics.json
```

---

## Part 3: MLflow Integration

### Objective
Track experiments, parameters, and metrics using MLflow.

### Step 3.1: Navigate to Part3

```powershell
cd ..\Part3
```

### Step 3.2: Start MLflow UI (Optional - for viewing during training)

**Open a NEW terminal window** and run:
```powershell
cd "C:\Users\ashmitad\Documents\Personal\Bits\SEMESTER 3\MLOPs\Assignment 1\MLOP-Assign\Part3"
mlflow ui --port 5000
```

**Keep this terminal open.** Access MLflow UI at: http://localhost:5000

### Step 3.3: Train Models with MLflow Tracking

**In your original terminal:**
```powershell
python src\train_with_mlflow.py
```

**Expected Output:**
```
Starting MLflow experiment: Heart Disease Classification
MLflow tracking URI: file:///.../Part3/mlruns

Training Logistic Regression with MLflow...
  Logged params: C, penalty, solver, class_weight
  Logged metrics: accuracy_mean, precision_mean, recall_mean, roc_auc_mean
  Logged artifacts: confusion matrix, ROC curve, metrics JSON
  ✓ Run ID: abc123...

Training Random Forest with MLflow...
  Logged params: n_estimators, max_depth, min_samples_leaf
  Logged metrics: accuracy_mean, precision_mean, recall_mean, roc_auc_mean
  Logged artifacts: confusion matrix, ROC curve, metrics JSON
  ✓ Run ID: def456...

✓ Experiment completed!
✓ View results at: http://localhost:5000
```

**Training Time:** 3-7 minutes.

### Step 3.4: View MLflow Experiments

1. Open browser: **http://localhost:5000**
2. You should see experiment: **"Heart Disease Classification"**
3. Click on the experiment to see all runs
4. Compare models by:
   - Selecting multiple runs
   - Clicking **"Compare"**
   - Viewing metrics side-by-side

### Step 3.5: Explore MLflow Features

**View Run Details:**
1. Click on any run
2. See **Parameters** (hyperparameters used)
3. See **Metrics** (performance metrics)
4. See **Artifacts** (plots, models, reports)

**Compare Models:**
1. Select both Logistic Regression and Random Forest runs
2. Click **"Compare"**
3. View parallel coordinates plot
4. View metric comparison table

### Step 3.6: Verify MLflow Outputs

```powershell
# Check MLflow directory structure
ls mlruns\

# Check outputs (same as Part2)
ls outputs\models\
ls outputs\metrics\
ls outputs\plots\
```

### Step 3.7: Register Best Model (Optional)

In MLflow UI:
1. Click on the best performing run (highest ROC-AUC)
2. Scroll to **Artifacts** section
3. Click on the model folder
4. Click **"Register Model"**
5. Create new model: "heart-disease-classifier"
6. Click **"Register"**

---

## Part 4: Model Packaging & Inference

### Objective
Package the best model and create an inference pipeline.

### Step 4.1: Navigate to Part4

```powershell
cd ..\Part4
```

### Step 4.2: Package the Model

```powershell
python src\package_model.py
```

**Expected Output:**
```
Packaging best model from Part3...
✓ Loaded model: RandomForestClassifier
✓ Model saved to: models/final_model.joblib
✓ MLflow model saved to: models/mlflow_model/
✓ Schema saved to: models/schema.json
✓ Metadata saved to: metrics/final_report.json

Model packaging complete!
```

### Step 4.3: Verify Packaged Model

```powershell
ls models\
ls models\mlflow_model\
```

**Expected Files:**
- `models/final_model.joblib` - Serialized model
- `models/mlflow_model/` - MLflow format model
- `models/schema.json` - Input schema
- `metrics/final_report.json` - Model metadata

### Step 4.4: Test Inference

```powershell
python src\infer.py
```

**Expected Output:**
```
Loading model from: models/final_model.joblib
✓ Model loaded successfully

Loading test data from: inputs/sample_X.csv
✓ Loaded 10 samples

Running inference...
✓ Predictions completed

Sample predictions:
  Patient 1: Class 0 (Probability: 0.65)
  Patient 2: Class 1 (Probability: 0.78)
  ...

✓ Predictions saved to: outputs/predictions.csv
```

### Step 4.5: View Predictions

```powershell
Get-Content outputs\predictions.csv
```

### Step 4.6: Custom Inference Test

Create a test file `test_patient.csv`:
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
63,1,1,145,233,1,2,150,0,2.3,3,0,6
```

Run inference:
```powershell
python src\infer.py --input test_patient.csv --output my_predictions.csv
```

---

## Part 5: Testing (Optional)

### Objective
Run unit tests to verify code quality.

### Step 5.1: Navigate to Part5

```powershell
cd ..\Part5
```

### Step 5.2: Run Tests

```powershell
pytest tests\ -v
```

**Expected Output:**
```
tests/test_data_preprocessing.py::test_load_data PASSED
tests/test_features.py::test_feature_engineering PASSED
tests/test_model.py::test_model_training PASSED
tests/test_inference.py::test_inference PASSED

========== 4 passed in 2.34s ==========
```

### Step 5.3: Run Tests with Coverage

```powershell
pytest tests\ --cov=src --cov-report=html
```

View coverage report:
```powershell
Start-Process htmlcov\index.html
```

---

## Part 6: API Development

### Objective
Create a FastAPI REST API for the model.

### Step 6.1: Navigate to Part6

```powershell
cd ..\Part6
```

### Step 6.2: Test API Locally (Before Docker)

```powershell
python src\app.py
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6.3: Test API Endpoints

**Open a NEW terminal** and test:

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Prediction
$body = @{
    age=63; sex=1; cp=1; trestbps=145; chol=233; 
    fbs=1; restecg=2; thalach=150; exang=0; 
    oldpeak=2.3; slope=3; ca=0; thal=6
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

### Step 6.4: View API Documentation

Open browser: **http://localhost:8000/docs**

### Step 6.5: Stop Local API

Press `Ctrl+C` in the terminal running the API.

### Step 6.6: Build Docker Image

```powershell
cd ..
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
```

**Build Time:** 3-5 minutes.

### Step 6.7: Test Docker Container

```powershell
docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest
```

Test:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

Stop container:
```powershell
docker stop heart-api
docker rm heart-api
```

---

## Part 7 & 8: Deployment & Monitoring

### Objective
Deploy to Kubernetes (Minikube) and set up monitoring with Prometheus & Grafana.

### ✅ Continue with Existing Documentation

**Now proceed to:**
- **Part8/DEPLOYMENT_STEPS.md** for complete Step 7 & 8 instructions

The Docker image you built in Part 6 will be used for:
- **Step 7:** Kubernetes deployment on Minikube
- **Step 8:** Monitoring stack with Docker Compose

---

## Complete Workflow Summary

```
Part 1: Data Preprocessing
  ↓
Part 2: Model Training (Logistic Regression + Random Forest)
  ↓
Part 3: MLflow Tracking (Experiment tracking + Model registry)
  ↓
Part 4: Model Packaging (Best model → Production format)
  ↓
Part 5: Testing (Unit tests + Coverage)
  ↓
Part 6: API Development (FastAPI + Docker)
  ↓
Part 7: Kubernetes Deployment (Minikube + LoadBalancer)
  ↓
Part 8: Monitoring (Prometheus + Grafana + Logging)
```

---

## Verification Checklist

### Part 1 ✓
- [ ] Raw data downloaded
- [ ] Clean data created (`heart_clean.csv`)
- [ ] Encoded data created (`heart_encoded.csv`)
- [ ] EDA plots generated (3 figures)

### Part 2 ✓
- [ ] Logistic Regression trained
- [ ] Random Forest trained
- [ ] Models saved (`.joblib` files)
- [ ] Metrics saved (JSON files)
- [ ] Plots generated (confusion matrix, ROC curves)

### Part 3 ✓
- [ ] MLflow UI accessible
- [ ] Experiments tracked
- [ ] Parameters logged
- [ ] Metrics logged
- [ ] Artifacts saved
- [ ] Models registered (optional)

### Part 4 ✓
- [ ] Best model packaged
- [ ] MLflow model format created
- [ ] Schema saved
- [ ] Inference tested
- [ ] Predictions generated

### Part 5 ✓ (Optional)
- [ ] All tests pass
- [ ] Coverage report generated

### Part 6 ✓
- [ ] API runs locally
- [ ] API documentation accessible
- [ ] Docker image built
- [ ] Docker container tested

### Part 7 & 8 ✓
- [ ] See `Part8/DEPLOYMENT_STEPS.md` for checklist

---

## Common Issues & Solutions

### Issue: Module not found errors
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: MLflow UI not starting
```powershell
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Use different port
mlflow ui --port 5001
```

### Issue: Python script can't find modules
```powershell
# Run from project root
cd "C:\Users\ashmitad\Documents\Personal\Bits\SEMESTER 3\MLOPs\Assignment 1\MLOP-Assign"

# Or add to PYTHONPATH
$env:PYTHONPATH = (Get-Location).Path
```

### Issue: Docker build fails
```powershell
# Clean Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t heart-disease-api:latest -f Part6/Dockerfile .
```

---

## Quick Command Reference

### Virtual Environment
```powershell
# Activate
.\venv\Scripts\Activate.ps1

# Deactivate
deactivate
```

### Run All Parts Sequentially
```powershell
# Part 1
cd Part1
python dataset_download_script\download_data.py
python src\data_preprocess.py
python src\eda.py

# Part 2
cd ..\Part2
python src\train_models.py

# Part 3
cd ..\Part3
python src\train_with_mlflow.py

# Part 4
cd ..\Part4
python src\package_model.py
python src\infer.py

# Part 6
cd ..\Part6
python src\app.py
```

### MLflow Commands
```powershell
# Start UI
mlflow ui --port 5000

# Start UI with specific backend
mlflow ui --backend-store-uri file:///path/to/mlruns --port 5000

# View experiments
mlflow experiments list
```

---

## Time Estimates

| Part | Task | Estimated Time |
|------|------|----------------|
| Part 1 | Data preprocessing & EDA | 5-10 minutes |
| Part 2 | Model training | 5-10 minutes |
| Part 3 | MLflow integration | 10-15 minutes |
| Part 4 | Model packaging | 3-5 minutes |
| Part 5 | Testing (optional) | 5-10 minutes |
| Part 6 | API development | 10-15 minutes |
| Part 7 | Kubernetes deployment | 15-20 minutes |
| Part 8 | Monitoring setup | 10-15 minutes |
| **Total** | | **60-100 minutes** |

---

## Next Steps

1. ✅ Complete Parts 1-6 using this guide
2. ✅ Proceed to `Part8/DEPLOYMENT_STEPS.md` for Steps 7 & 8
3. ✅ Follow `Part8/SCREENSHOT_GUIDE.md` for documentation
4. ✅ Use `Part8/POWERSHELL_COMMANDS.md` for testing

---

**Good luck with your MLOps assignment! 🚀**

For questions or issues, refer to the troubleshooting sections in each part's README or the main DEPLOYMENT_STEPS.md file.

