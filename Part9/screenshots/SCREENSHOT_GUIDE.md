# Screenshot Guide
## Required Screenshots for Documentation

This guide lists all screenshots needed for the final submission.

---

## 1. Data & EDA Screenshots

### 1.1 Raw Data
**File**: `01_raw_data.png`
**Command**: 
```bash
head -10 Part1/data/raw/processed.cleveland.data
```
**What to capture**: First 10 lines of raw data showing original format

### 1.2 Cleaned Data
**File**: `02_cleaned_data.png`
**Command**:
```bash
head -10 Part1/data/interim/heart_clean.csv
```
**What to capture**: Cleaned data with proper column names

### 1.3 Histograms
**File**: `03_histograms.png`
**Location**: `Part1/reports/figures/histograms_numeric.png`
**What to capture**: Distribution of numeric features

### 1.4 Class Balance
**File**: `04_class_balance.png`
**Location**: `Part1/reports/figures/class_balance.png`
**What to capture**: Bar chart showing target distribution

### 1.5 Correlation Heatmap
**File**: `05_correlation_heatmap.png`
**Location**: `Part1/reports/figures/corr_heatmap.png`
**What to capture**: Correlation matrix with target

---

## 2. Model Training Screenshots

### 2.1 Training Output
**File**: `06_model_training.png`
**Command**:
```bash
python Part2/src/train_models.py
```
**What to capture**: Terminal output showing training progress and metrics

### 2.2 Model Metrics Summary
**File**: `07_model_metrics.png`
**Command**:
```bash
cat Part2/outputs/metrics/scores_summary.csv
```
**What to capture**: CSV showing comparison of both models

### 2.3 ROC Curve - Logistic Regression
**File**: `08_logreg_roc.png`
**Location**: `Part2/outputs/plots/logreg_roc_curve.png`
**What to capture**: ROC curve for Logistic Regression

### 2.4 ROC Curve - Random Forest
**File**: `09_rf_roc.png`
**Location**: `Part2/outputs/plots/rf_roc_curve.png`
**What to capture**: ROC curve for Random Forest

### 2.5 Confusion Matrix
**File**: `10_confusion_matrix.png`
**Location**: `Part2/outputs/plots/rf_confusion_matrix.png`
**What to capture**: Confusion matrix for selected model

---

## 3. MLflow Screenshots

### 3.1 MLflow UI - Experiments List
**File**: `11_mlflow_experiments.png`
**URL**: http://localhost:5000
**What to capture**: List of experiments showing all runs

### 3.2 MLflow UI - Run Details
**File**: `12_mlflow_run_details.png`
**URL**: http://localhost:5000 (click on a run)
**What to capture**: Detailed view of a single run showing parameters and metrics

### 3.3 MLflow UI - Metrics Comparison
**File**: `13_mlflow_metrics_comparison.png`
**URL**: http://localhost:5000 (compare runs)
**What to capture**: Side-by-side comparison of multiple runs

### 3.4 MLflow UI - Artifacts
**File**: `14_mlflow_artifacts.png`
**URL**: http://localhost:5000 (artifacts tab)
**What to capture**: Logged artifacts (plots, models, reports)

### 3.5 MLflow UI - Model Registry
**File**: `15_mlflow_model_registry.png`
**URL**: http://localhost:5000 (models tab)
**What to capture**: Registered models with versions

---

## 4. Testing Screenshots

### 4.1 Test Execution
**File**: `16_pytest_execution.png`
**Command**:
```bash
cd Part5
pytest -v
```
**What to capture**: All tests passing with green checkmarks

### 4.2 Test Coverage Report
**File**: `17_test_coverage.png`
**Command**:
```bash
cd Part5
pytest --cov --cov-report=term
```
**What to capture**: Coverage percentage for each module

### 4.3 Coverage HTML Report
**File**: `18_coverage_html.png`
**Location**: Open `Part5/htmlcov/index.html` in browser
**What to capture**: HTML coverage report showing detailed coverage

---

## 5. CI/CD Screenshots

### 5.1 GitHub Actions - Workflow List
**File**: `19_github_actions_list.png`
**URL**: GitHub repository → Actions tab
**What to capture**: List of workflow runs with statuses

### 5.2 GitHub Actions - Workflow Details
**File**: `20_github_actions_details.png`
**URL**: Click on a workflow run
**What to capture**: Detailed view showing all jobs (lint, test, train)

### 5.3 GitHub Actions - Job Logs
**File**: `21_github_actions_logs.png`
**URL**: Click on a job to see logs
**What to capture**: Logs from testing or training job

### 5.4 GitHub Actions - Artifacts
**File**: `22_github_actions_artifacts.png`
**URL**: Scroll down on workflow run page
**What to capture**: List of uploaded artifacts (models, metrics, plots)

---

## 6. API Screenshots

### 6.1 API Root Endpoint
**File**: `23_api_root.png`
**URL**: http://localhost:8000
**What to capture**: JSON response from root endpoint

### 6.2 API Health Check
**File**: `24_api_health.png`
**URL**: http://localhost:8000/health
**What to capture**: Health check response showing status

### 6.3 API Documentation (Swagger)
**File**: `25_api_swagger.png`
**URL**: http://localhost:8000/docs
**What to capture**: Swagger UI showing all endpoints

### 6.4 API Prediction Request
**File**: `26_api_prediction_request.png`
**URL**: http://localhost:8000/docs (try /predict endpoint)
**What to capture**: Request body with sample patient data

### 6.5 API Prediction Response
**File**: `27_api_prediction_response.png`
**URL**: http://localhost:8000/docs (after executing)
**What to capture**: Response showing prediction, probability, risk level

### 6.6 API Model Info
**File**: `28_api_model_info.png`
**URL**: http://localhost:8000/model/info
**What to capture**: Model metadata and performance metrics

### 6.7 API Batch Prediction
**File**: `29_api_batch_prediction.png`
**URL**: http://localhost:8000/docs (try /batch_predict)
**What to capture**: Batch prediction with multiple patients

---

## 7. Docker Screenshots

### 7.1 Docker Build
**File**: `30_docker_build.png`
**Command**:
```bash
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
```
**What to capture**: Successful build output showing all layers

### 7.2 Docker Images List
**File**: `31_docker_images.png`
**Command**:
```bash
docker images | grep heart-disease
```
**What to capture**: Image listed with size and creation date

### 7.3 Docker Container Running
**File**: `32_docker_ps.png`
**Command**:
```bash
docker ps
```
**What to capture**: Container running with port mapping

### 7.4 Docker Logs
**File**: `33_docker_logs.png`
**Command**:
```bash
docker logs heart-disease-api
```
**What to capture**: Container logs showing API startup

### 7.5 Docker Container Stats
**File**: `34_docker_stats.png`
**Command**:
```bash
docker stats heart-disease-api --no-stream
```
**What to capture**: CPU and memory usage

---

## 8. Kubernetes Screenshots

### 8.1 Kubernetes Deployment
**File**: `35_k8s_deployment.png`
**Command**:
```bash
kubectl get deployments -n mlops-heart-disease
```
**What to capture**: Deployment status showing replicas

### 8.2 Kubernetes Pods
**File**: `36_k8s_pods.png`
**Command**:
```bash
kubectl get pods -n mlops-heart-disease
```
**What to capture**: All pods running with status

### 8.3 Kubernetes Services
**File**: `37_k8s_services.png`
**Command**:
```bash
kubectl get services -n mlops-heart-disease
```
**What to capture**: Service with LoadBalancer and external IP

### 8.4 Kubernetes HPA
**File**: `38_k8s_hpa.png`
**Command**:
```bash
kubectl get hpa -n mlops-heart-disease
```
**What to capture**: HPA showing current/desired replicas and metrics

### 8.5 Kubernetes Pod Logs
**File**: `39_k8s_logs.png`
**Command**:
```bash
kubectl logs -f deployment/heart-disease-api -n mlops-heart-disease --tail=20
```
**What to capture**: Recent logs from pods

### 8.6 Kubernetes Describe Pod
**File**: `40_k8s_describe.png`
**Command**:
```bash
kubectl describe pod <pod-name> -n mlops-heart-disease
```
**What to capture**: Detailed pod information including events

---

## 9. Monitoring Screenshots

### 9.1 Prometheus Targets
**File**: `41_prometheus_targets.png`
**URL**: http://localhost:9090/targets
**What to capture**: All targets showing as UP

### 9.2 Prometheus Metrics
**File**: `42_prometheus_metrics.png`
**URL**: http://localhost:8000/metrics
**What to capture**: Raw Prometheus metrics from API

### 9.3 Prometheus Query
**File**: `43_prometheus_query.png`
**URL**: http://localhost:9090/graph
**Query**: `rate(api_requests_total[5m])`
**What to capture**: Graph showing request rate over time

### 9.4 Grafana Login
**File**: `44_grafana_login.png`
**URL**: http://localhost:3000
**What to capture**: Grafana login page

### 9.5 Grafana Data Source
**File**: `45_grafana_datasource.png`
**URL**: http://localhost:3000 (Configuration → Data Sources)
**What to capture**: Prometheus data source configured

### 9.6 Grafana Dashboard - Overview
**File**: `46_grafana_dashboard_overview.png`
**URL**: http://localhost:3000 (dashboards)
**What to capture**: Full dashboard showing all panels

### 9.7 Grafana Dashboard - Request Rate
**File**: `47_grafana_request_rate.png`
**URL**: http://localhost:3000 (zoom into request rate panel)
**What to capture**: Request rate graph with data

### 9.8 Grafana Dashboard - Latency
**File**: `48_grafana_latency.png`
**URL**: http://localhost:3000 (zoom into latency panel)
**What to capture**: Latency percentiles graph

### 9.9 Grafana Dashboard - Predictions
**File**: `49_grafana_predictions.png`
**URL**: http://localhost:3000 (zoom into predictions panel)
**What to capture**: Prediction distribution by risk level

### 9.10 Grafana Dashboard - Errors
**File**: `50_grafana_errors.png`
**URL**: http://localhost:3000 (zoom into errors panel)
**What to capture**: Error rate over time

---

## 10. Additional Screenshots

### 10.1 Project Structure
**File**: `51_project_structure.png`
**Command**:
```bash
tree -L 2 -d
# or
ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/'
```
**What to capture**: Directory tree showing all parts

### 10.2 Requirements File
**File**: `52_requirements.png`
**Command**:
```bash
cat requirements.txt
```
**What to capture**: All dependencies listed

### 10.3 Final Model Files
**File**: `53_model_files.png`
**Command**:
```bash
ls -lh Part4/models/
```
**What to capture**: Final model files with sizes

### 10.4 Test API Script Output
**File**: `54_test_api_output.png`
**Command**:
```bash
cd Part6/src
python test_api.py
```
**What to capture**: All API tests passing

---

## Screenshot Organization

### Folder Structure
```
Part9/screenshots/
├── 01_data_eda/
│   ├── 01_raw_data.png
│   ├── 02_cleaned_data.png
│   ├── 03_histograms.png
│   ├── 04_class_balance.png
│   └── 05_correlation_heatmap.png
├── 02_model_training/
│   ├── 06_model_training.png
│   ├── 07_model_metrics.png
│   ├── 08_logreg_roc.png
│   ├── 09_rf_roc.png
│   └── 10_confusion_matrix.png
├── 03_mlflow/
│   ├── 11_mlflow_experiments.png
│   ├── 12_mlflow_run_details.png
│   ├── 13_mlflow_metrics_comparison.png
│   ├── 14_mlflow_artifacts.png
│   └── 15_mlflow_model_registry.png
├── 04_testing/
│   ├── 16_pytest_execution.png
│   ├── 17_test_coverage.png
│   └── 18_coverage_html.png
├── 05_cicd/
│   ├── 19_github_actions_list.png
│   ├── 20_github_actions_details.png
│   ├── 21_github_actions_logs.png
│   └── 22_github_actions_artifacts.png
├── 06_api/
│   ├── 23_api_root.png
│   ├── 24_api_health.png
│   ├── 25_api_swagger.png
│   ├── 26_api_prediction_request.png
│   ├── 27_api_prediction_response.png
│   ├── 28_api_model_info.png
│   └── 29_api_batch_prediction.png
├── 07_docker/
│   ├── 30_docker_build.png
│   ├── 31_docker_images.png
│   ├── 32_docker_ps.png
│   ├── 33_docker_logs.png
│   └── 34_docker_stats.png
├── 08_kubernetes/
│   ├── 35_k8s_deployment.png
│   ├── 36_k8s_pods.png
│   ├── 37_k8s_services.png
│   ├── 38_k8s_hpa.png
│   ├── 39_k8s_logs.png
│   └── 40_k8s_describe.png
├── 09_monitoring/
│   ├── 41_prometheus_targets.png
│   ├── 42_prometheus_metrics.png
│   ├── 43_prometheus_query.png
│   ├── 44_grafana_login.png
│   ├── 45_grafana_datasource.png
│   ├── 46_grafana_dashboard_overview.png
│   ├── 47_grafana_request_rate.png
│   ├── 48_grafana_latency.png
│   ├── 49_grafana_predictions.png
│   └── 50_grafana_errors.png
└── 10_additional/
    ├── 51_project_structure.png
    ├── 52_requirements.png
    ├── 53_model_files.png
    └── 54_test_api_output.png
```

### Screenshot Tips

1. **Resolution**: Minimum 1920x1080
2. **Format**: PNG (lossless)
3. **Annotations**: Add arrows or highlights if needed
4. **Cropping**: Remove unnecessary parts but keep context
5. **Naming**: Use descriptive names as listed above
6. **Quality**: Ensure text is readable
7. **Timestamps**: Include if relevant (logs, metrics)
8. **Consistency**: Use same terminal theme/colors

### Tools for Screenshots

- **Windows**: Snipping Tool, Win+Shift+S
- **macOS**: Cmd+Shift+4
- **Linux**: gnome-screenshot, flameshot
- **Browser**: Built-in developer tools screenshot feature

---

## Checklist

Before submission, verify you have:

- [ ] All 54 screenshots captured
- [ ] Screenshots are organized in folders
- [ ] All screenshots are readable (text not blurry)
- [ ] File names match this guide
- [ ] Screenshots show successful operations
- [ ] Timestamps are visible where relevant
- [ ] No sensitive information visible (API keys, passwords)
- [ ] Screenshots are in PNG format
- [ ] Resolution is adequate (min 1920x1080)

---

**Total Screenshots**: 54  
**Estimated Time**: 2-3 hours to capture all

**Note**: Some screenshots (especially Kubernetes and monitoring) may require the respective systems to be running. Plan accordingly.

