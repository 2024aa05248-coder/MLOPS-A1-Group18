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
- [Part 7: Deploy to Minikube (Kubernetes)](#part-7-deploy-to-minikube-kubernetes)
- [Part 8: Full Monitoring Stack (Docker Compose)](#part-8-full-monitoring-stack-docker-compose)
- [Monitoring Metrics Available](#monitoring-metrics-available)
- [Cleanup Commands](#cleanup-commands)
- [Troubleshooting](#troubleshooting)
- [Key Files](#key-files)
- [Success Indicators](#success-indicators)

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
cd "<path-to-your-project-directory>"
# Example: cd "C:\Projects\MLOPS-A1-Group18"
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

### Step 1.2: Data Preprocessing

```powershell
python src\data_preprocess.py
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
cd <PROJECT_DIRECTORY>\Part3
# Example: cd C:\Projects\MLOPS-A1-Group18\Part3
mlflow ui --port 5000
```

**Keep this terminal open.** Access MLflow UI at: http://localhost:5000

### Step 3.3: Train Models with MLflow Tracking

**In your original terminal:**
```powershell
python src\train_with_mlflow.py
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
python src\infer.py --input inputs\sample_X.csv
```

### Step 4.5: View Predictions

```powershell
Get-Content outputs\predictions.csv
```

### Step 4.6: Custom Inference Test

#### Create a test file `test_patient.csv`:
```powershell
"age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal`n63,1,1,145,233,1,2,150,0,2.3,3,0,6" | Out-File -FilePath test_patient.csv -Encoding utf8
```

Run inference:
```powershell
python src\infer.py --input test_patient.csv --output my_predictions.csv
```

---

## Part 5: Testing

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
```text
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

## Part 7: Deploy to Minikube (Kubernetes)

This part deploys the Heart Disease Prediction API to a local Kubernetes cluster (Minikube) using the Docker image built in Part 6.

### Step 7.1: Build the Docker Image (if not already built)

From the project root:

```powershell
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
```

### Step 7.2: Start Minikube

```powershell
minikube start --driver=docker
```

Wait 2-3 minutes for Minikube to fully start.

### Step 7.3: Load Docker Image into Minikube

```powershell
minikube image load heart-disease-api:latest
```

Verify:

```powershell
minikube image ls | findstr heart-disease-api
```

### Step 7.4: Create Kubernetes Deployment

Apply the deployment manifest:

```powershell
kubectl apply -f Part7/k8s/deployment.yaml
```

Expected output:
```text
deployment.apps/heart-disease-api created
service/heart-disease-api-service created
```

### Step 7.5: Verify Deployment

Check pods:

```powershell
kubectl get pods
```

After 1-2 minutes you should see all pods in `Running` state.

Check deployment:

```powershell
kubectl get deployment heart-disease-api
```

Check service:

```powershell
kubectl get service heart-disease-api-service
```

### Step 7.6: Expose Service via Minikube Tunnel

**Open a NEW terminal window** and run:

```powershell
minikube tunnel
```

Keep this terminal running.

### Step 7.7: Get Service URL

In your original terminal:

```powershell
kubectl get service heart-disease-api-service
```

Wait until `EXTERNAL-IP` is assigned.

Or:

```powershell
minikube service heart-disease-api-service --url
```

Keep this terminal open.

### Step 7.8: Test the Deployment

Replace `<PORT>` with the port from the previous command.

Health endpoint:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:<PORT>/health"
```

Root endpoint:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:<PORT>/"
```

Prediction:

```powershell
$body = @{
    age=63; sex=1; cp=1; trestbps=145; chol=233; 
    fbs=1; restecg=2; thalach=150; exang=0; 
    oldpeak=2.3; slope=3; ca=0; thal=6
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:<PORT>/predict" -Method Post -Body $body -ContentType "application/json"
```

### Step 7.9: Recommended Screenshots

1. `kubectl get pods` - all 3 pods `Running`
2. `kubectl get deployment` - deployment status
3. `kubectl get service` - service with `EXTERNAL-IP`
4. Browser at `http://<EXTERNAL-IP>/docs`
5. Successful prediction response (browser/Postman)
6. `kubectl describe deployment heart-disease-api`

---

## Part 8: Full Monitoring Stack (Docker Compose)

This part runs the API with Prometheus and Grafana using Docker Compose.

### Step 8.1: Stop Minikube (Optional)

To free resources:

```powershell
# In the terminal running minikube tunnel, press Ctrl+C
minikube stop
```

### Step 8.2: Navigate to Part8 Directory

```powershell
cd Part8
```

### Step 8.3: Start the Full Monitoring Stack

```powershell
docker-compose -f docker-compose-monitoring.yml up -d
```

This starts:
- API on port 8000
- Prometheus on port 9090
- Grafana on port 3000

### Step 8.4: Verify All Containers are Running

```powershell
docker-compose -f docker-compose-monitoring.yml ps
```

Check logs if needed:

```powershell
docker-compose -f docker-compose-monitoring.yml logs api
docker-compose -f docker-compose-monitoring.yml logs prometheus
docker-compose -f docker-compose-monitoring.yml logs grafana
```

### Step 8.5: Access and Test the API

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Root endpoint
Invoke-RestMethod -Uri "http://localhost:8000/"

# Metrics endpoint (Prometheus format)
Invoke-WebRequest -Uri "http://localhost:8000/metrics"

# Prediction
$body = @{
    age=63; sex=1; cp=1; trestbps=145; chol=233; 
    fbs=1; restecg=2; thalach=150; exang=0; 
    oldpeak=2.3; slope=3; ca=0; thal=6
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

### Step 8.6: Access Prometheus

1. Open **http://localhost:9090**
2. Go to **Status → Targets**
3. Verify API target is **UP**
4. Example queries:
   - `api_requests_total`
   - `api_request_duration_seconds_sum`
   - `predictions_total`
   - `rate(api_requests_total[5m])`
   - `prediction_duration_seconds_sum`

### Step 8.7: Access Grafana

1. Open **http://localhost:3000**
2. Login:
   - Username: `admin`
   - Password: `admin`
3. Skip or change password.

### Step 8.8: Configure Grafana Dashboard

**Verify Prometheus Data Source:**
1. Go to **Connections (gear icon) → Data Sources**
2. Select **Prometheus** (auto-provisioned)
3. Click **Save & Test**

**Import Dashboard:**
1. Go to **Dashboards → Browse → Import**
2. Click **Upload JSON file**
3. Select `dashboards/api-dashboard.json` from `Part8`
4. Choose **Prometheus** as the data source
5. Click **Import**

**Optional - Create Custom Dashboard:**
Use queries like:
- `rate(api_requests_total[5m])`
- `api_request_duration_seconds_sum / api_request_duration_seconds_count`
- `predictions_total`
- `active_requests`

### Step 8.9: Generate Traffic and View Metrics

```powershell
# Run 20 predictions
for ($i=1; $i -le 20; $i++) {
    $body = @{
        age=63; sex=1; cp=1; trestbps=145; chol=233; 
        fbs=1; restecg=2; thalach=150; exang=0; 
        oldpeak=2.3; slope=3; ca=0; thal=6
    } | ConvertTo-Json
    
    Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json" | Out-Null
    Write-Host "Request $i completed"
    Start-Sleep -Milliseconds 500
}
```

### Step 8.10: View Logs

```powershell
# Real-time logs
docker-compose -f docker-compose-monitoring.yml logs -f api

# Or from log file
Get-Content logs\api.log -Tail 50
```

Each request log includes:
- Timestamp
- HTTP method and endpoint
- Client IP
- Status code
- Duration
- Prediction details (`/predict`)

### Step 8.11: Recommended Screenshots

1. `docker-compose ps` - 3 containers up
2. API accessible at `http://localhost:8000/`
3. Metrics at `http://localhost:8000/metrics`
4. Prometheus at `http://localhost:9090`
5. Prometheus scraping API (Targets page)
6. Grafana at `http://localhost:3000`
7. Dashboard configured
8. Terminal showing API logs
9. Successful prediction responses

---

## Monitoring Metrics Available

### Request Metrics
- `api_requests_total{method, endpoint, status}` - Total requests
- `api_request_duration_seconds{method, endpoint}` - Duration histogram
- `active_requests` - Current active requests

### Prediction Metrics
- `predictions_total{prediction_class, risk_level}` - Total predictions
- `prediction_duration_seconds` - Prediction time histogram

### Model Metrics
- `model_load_time_seconds` - Model load time

### Error Metrics
- `api_errors_total{error_type, endpoint}` - Errors by type and endpoint

---

## Cleanup Commands

### Stop Monitoring Stack

```powershell
cd Part8
docker-compose -f docker-compose-monitoring.yml down
```

### Stop and Clean Monitoring Stack (including volumes)

```powershell
docker-compose -f docker-compose-monitoring.yml down -v
```

### Stop Minikube Deployment

```powershell
kubectl delete -f Part7/k8s/deployment.yaml
minikube stop
```

### Delete Minikube Cluster

```powershell
minikube delete
```

## Troubleshooting

### API Container Constantly Restarting

Fix path handling in `Part8/src/app_with_monitoring.py` (already applied in repo):
- Use correct `PROJECT_ROOT`
- Use `models/` instead of `Part4/models/`
- Use `metrics/` instead of `Part4/metrics/`

### Minikube Image Load Fails

```powershell
docker save heart-disease-api:latest | minikube image load -
```

### Pods Not Starting

```powershell
kubectl logs <pod-name>
kubectl describe pod <pod-name>
```

### Docker Compose Fails to Start

```powershell
docker-compose -f docker-compose-monitoring.yml logs <service-name>
docker-compose -f docker-compose-monitoring.yml build --no-cache
docker-compose -f docker-compose-monitoring.yml up -d
```

### Prometheus Can't Scrape API

- Check API container: `docker ps`
- Check `Part8/config/prometheus.yml`
- Ensure target is `api:8000`, not `localhost:8000`
- Check network: `docker network ls`
- Test `http://localhost:8000/metrics`

### Grafana Can't Connect to Prometheus

- Check Prometheus container: `docker ps`
- Datasource URL must be `http://prometheus:9090`
- Ensure both are on `monitoring` network

## Key Files

- `Part1/src/data_preprocess.py` - preprocessing
- `Part2/src/train_models.py` - baseline training
- `Part3/src/train_with_mlflow.py` - MLflow training
- `Part4/src/package_model.py` - packaging
- `Part4/src/infer.py` - batch inference
- `Part5/tests/` - unit tests
- `Part6/src/app.py` - FastAPI app
- `Part6/Dockerfile` - API container
- `Part7/k8s/deployment.yaml` - Kubernetes manifest
- `Part8/docker-compose-monitoring.yml` - monitoring stack
- `Part8/config/prometheus.yml` - Prometheus config
- `Part8/config/grafana-datasource.yml` - Grafana datasource
- `Part8/src/app_with_monitoring.py` - API with instrumentation

---

## Success Indicators

### Minikube Deployment

- 3 pods running in Kubernetes
- LoadBalancer service has external IP
- API responds to `/health`
- `/predict` returns expected JSON

### Monitoring

- 3 containers (API, Prometheus, Grafana) are healthy
- API logs each request
- Prometheus target is UP
- Metrics visible via queries and Grafana dashboard

Deployment and monitoring completed successfully.
