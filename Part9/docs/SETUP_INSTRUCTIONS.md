# Setup Instructions
## Heart Disease Prediction MLOps Project

This guide provides step-by-step instructions to set up and run the entire MLOps pipeline from scratch.

---

## Prerequisites

### Required Software
- **Python**: 3.9 or higher
- **Docker**: 20.10 or higher
- **Git**: Latest version
- **kubectl**: For Kubernetes deployment (optional)
- **Minikube** or **Docker Desktop** with Kubernetes enabled (optional)

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 10GB free space
- **CPU**: 4 cores recommended

---

## Part 1: Environment Setup

### 1.1 Clone Repository

```bash
git clone <repository-url>
cd MLOP-Assign
```

### 1.2 Create Virtual Environment

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify Installation**:
```bash
python -c "import sklearn, pandas, numpy, mlflow, fastapi; print('All packages installed successfully')"
```

---

## Part 2: Data Acquisition & Preprocessing

### 2.1 Download Dataset

```bash
python Part1/dataset_download_script/download_data.py
```

**Expected Output**: `Part1/data/raw/processed.cleveland.data`

### 2.2 Preprocess Data

```bash
python Part1/src/data_preprocess.py
```

**Expected Outputs**:
- `Part1/data/interim/heart_clean.csv`
- `Part1/data/processed/heart_encoded.csv`
- `Part1/data/processed/feature_names.json`

### 2.3 Run EDA

```bash
python Part1/src/eda.py
```

**Expected Outputs**: Plots in `Part1/reports/figures/`

---

## Part 3: Model Training

### 3.1 Train Models (Part 2)

```bash
python Part2/src/train_models.py
```

**Expected Outputs**:
- Models: `Part2/outputs/models/*.joblib`
- Metrics: `Part2/outputs/metrics/*.json`
- Plots: `Part2/outputs/plots/*.png`

**Duration**: ~2-5 minutes

### 3.2 Train with MLflow (Part 3)

```bash
python Part3/src/train_with_mlflow.py
```

**Expected Outputs**:
- MLflow runs: `Part3/mlruns/`
- Models and metrics in Part3/outputs/

### 3.3 View MLflow UI

```bash
mlflow ui --backend-store-uri file://$(pwd)/Part3/mlruns
```

**Access**: http://localhost:5000

### 3.4 Package Final Model (Part 4)

```bash
python Part4/src/package_model.py
```

**Expected Outputs**:
- `Part4/models/final_model.joblib`
- `Part4/models/mlflow_model/`
- `Part4/metrics/final_report.json`

---

## Part 4: Testing

### 4.1 Run Unit Tests

```bash
cd Part5
pytest -v
```

**Expected**: 40 tests should pass

### 4.2 Run with Coverage

```bash
pytest --cov=../Part1 --cov=../Part2 --cov=../Part4 --cov-report=html
```

**View Coverage**: Open `htmlcov/index.html` in browser

---

## Part 5: API Deployment

### 5.1 Run API Locally

```bash
cd Part6/src
python app.py
```

**Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### 5.2 Test API

**In another terminal**:
```bash
cd Part6/src
python test_api.py
```

Or use curl:
```bash
curl http://localhost:8000/health
```

---

## Part 6: Docker Containerization

### 6.1 Build Docker Image

```bash
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
```

**Duration**: ~5-10 minutes (first build)

### 6.2 Run Container

```bash
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

### 6.3 Verify Container

```bash
docker ps
docker logs heart-disease-api
curl http://localhost:8000/health
```

### 6.4 Stop Container

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

---

## Part 7: Kubernetes Deployment

### 7.1 Start Minikube (Local)

```bash
minikube start --cpus=4 --memory=8192
minikube addons enable metrics-server
```

### 7.2 Deploy to Kubernetes

```bash
cd Part7
chmod +x deploy.sh  # macOS/Linux only
./deploy.sh
```

**Windows**: Run commands from deploy.sh manually

### 7.3 Access API

**Minikube**:
```bash
minikube service heart-disease-api-service -n mlops-heart-disease --url
```

**Port Forward**:
```bash
kubectl port-forward service/heart-disease-api-service 8000:80 -n mlops-heart-disease
```

### 7.4 Monitor Deployment

```bash
kubectl get pods -n mlops-heart-disease
kubectl get services -n mlops-heart-disease
kubectl logs -f deployment/heart-disease-api -n mlops-heart-disease
```

### 7.5 Cleanup

```bash
cd Part7
./undeploy.sh
minikube stop
```

---

## Part 8: Monitoring Setup

### 8.1 Run with Monitoring

```bash
cd Part8
docker-compose -f docker-compose-monitoring.yml up -d
```

### 8.2 Access Monitoring

- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 8.3 Setup Grafana

1. Login to Grafana (admin/admin)
2. Change password
3. Go to Configuration → Data Sources
4. Add Prometheus: http://prometheus:9090
5. Import dashboard from `Part8/dashboards/api-dashboard.json`

### 8.4 Stop Monitoring

```bash
docker-compose -f docker-compose-monitoring.yml down
```

---

## Part 9: CI/CD Setup

### 9.1 GitHub Actions

1. Push code to GitHub repository
2. GitHub Actions will automatically run on push to main
3. View workflow runs in Actions tab

### 9.2 Local CI Simulation

**Linting**:
```bash
flake8 Part1 Part2 Part3 Part4 Part5 --max-line-length=120 --extend-ignore=E203,W503
```

**Testing**:
```bash
cd Part5
pytest -v
```

**Full Pipeline**:
```bash
# Run all steps from .github/workflows/ci-cd.yml manually
```

---

## Troubleshooting

### Issue: Model Not Found

**Solution**:
```bash
# Ensure Part 4 is completed
python Part4/src/package_model.py
ls Part4/models/final_model.joblib
```

### Issue: Port Already in Use

**Solution**:
```bash
# Find process using port 8000
# Windows: netstat -ano | findstr :8000
# macOS/Linux: lsof -i :8000

# Kill process or use different port
docker run -p 8080:8000 heart-disease-api:latest
```

### Issue: Docker Build Fails

**Solution**:
```bash
# Clean Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t heart-disease-api:latest -f Part6/Dockerfile .
```

### Issue: Kubernetes Pods Not Starting

**Solution**:
```bash
# Check pod status
kubectl describe pod <pod-name> -n mlops-heart-disease

# Check logs
kubectl logs <pod-name> -n mlops-heart-disease

# For Minikube, ensure image is loaded
minikube image load heart-disease-api:latest
```

### Issue: Tests Failing

**Solution**:
```bash
# Ensure data is preprocessed
python Part1/src/data_preprocess.py

# Run tests with verbose output
cd Part5
pytest -v -s

# Run specific test
pytest tests/test_model.py::TestModelTraining::test_logistic_regression_training -v
```

---

## Quick Start (All Steps)

For experienced users, run all steps in sequence:

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Data & Training
python Part1/dataset_download_script/download_data.py
python Part1/src/data_preprocess.py
python Part1/src/eda.py
python Part2/src/train_models.py
python Part3/src/train_with_mlflow.py
python Part4/src/package_model.py

# 3. Testing
cd Part5 && pytest -v && cd ..

# 4. Docker
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
docker run -d -p 8000:8000 heart-disease-api:latest

# 5. Test API
curl http://localhost:8000/health
```

---

## Next Steps

After setup:
1. Explore MLflow UI to view experiment runs
2. Test API endpoints using Swagger docs
3. Deploy to Kubernetes for production
4. Set up monitoring with Prometheus and Grafana
5. Customize and extend the pipeline

---

## Support

For issues or questions:
1. Check troubleshooting section
2. Review individual Part READMEs
3. Check logs for error messages
4. Refer to Final Report for detailed documentation

---

**Last Updated**: December 2025

