# Quick Reference Guide
## Heart Disease Prediction MLOps Project

**Quick commands and paths for common tasks**

---

## 🚀 Quick Start Commands

### Complete Pipeline (Fresh Setup)
```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run complete pipeline
python Part1/dataset_download_script/download_data.py
python Part1/src/data_preprocess.py
python Part1/src/eda.py
python Part2/src/train_models.py
python Part3/src/train_with_mlflow.py
python Part4/src/package_model.py

# Test
cd Part5 && pytest -v && cd ..

# Run API
cd Part6/src && python app.py
```

---

## 📂 Important File Locations

### Data Files
```
Part1/data/raw/processed.cleveland.data          # Raw data
Part1/data/interim/heart_clean.csv               # Cleaned data
Part1/data/processed/heart_encoded.csv           # Encoded features
```

### Models
```
Part2/outputs/models/logreg_best.joblib          # Logistic Regression
Part2/outputs/models/rf_best.joblib              # Random Forest
Part4/models/final_model.joblib                  # Final packaged model
Part4/models/mlflow_model/                       # MLflow model directory
```

### Metrics
```
Part2/outputs/metrics/scores_summary.csv         # Model comparison
Part4/metrics/final_report.json                  # Final model report
```

### Visualizations
```
Part1/reports/figures/histograms_numeric.png     # Feature distributions
Part1/reports/figures/class_balance.png          # Target distribution
Part1/reports/figures/corr_heatmap.png           # Correlation matrix
Part2/outputs/plots/rf_roc_curve.png             # ROC curve
Part2/outputs/plots/rf_confusion_matrix.png      # Confusion matrix
```

---

## 🧪 Testing Commands

```bash
# Run all tests
cd Part5 && pytest -v

# Run with coverage
pytest --cov=../Part1 --cov=../Part2 --cov=../Part4 --cov-report=html

# Run specific test file
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py::TestModelTraining::test_logistic_regression_training -v
```

---

## 🐳 Docker Commands

```bash
# Build image
docker build -t heart-disease-api:latest -f Part6/Dockerfile .

# Run container
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest

# View logs
docker logs -f heart-disease-api

# Stop and remove
docker stop heart-disease-api
docker rm heart-disease-api

# Run with monitoring
cd Part8
docker-compose -f docker-compose-monitoring.yml up -d
docker-compose -f docker-compose-monitoring.yml down
```

---

## ☸️ Kubernetes Commands

```bash
# Deploy
cd Part7 && ./deploy.sh

# Check status
kubectl get pods -n mlops-heart-disease
kubectl get services -n mlops-heart-disease
kubectl get hpa -n mlops-heart-disease

# View logs
kubectl logs -f deployment/heart-disease-api -n mlops-heart-disease

# Port forward
kubectl port-forward service/heart-disease-api-service 8000:80 -n mlops-heart-disease

# Undeploy
cd Part7 && ./undeploy.sh
```

---

## 🔍 API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints
```bash
# Root
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'

# Metrics (Prometheus)
curl http://localhost:8000/metrics

# API Documentation
# Open in browser: http://localhost:8000/docs
```

---

## 📊 MLflow Commands

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/Part3/mlruns

# Access UI
# Open in browser: http://localhost:5000
```

---

## 📈 Monitoring URLs

```
API:        http://localhost:8000
API Docs:   http://localhost:8000/docs
MLflow:     http://localhost:5000
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000  (admin/admin)
```

---

## 🔧 Troubleshooting Commands

### Check if model exists
```bash
ls -lh Part4/models/final_model.joblib
```

### Check if data is processed
```bash
ls -lh Part1/data/interim/heart_clean.csv
```

### Check Docker status
```bash
docker ps
docker images | grep heart-disease
```

### Check Kubernetes status
```bash
kubectl get all -n mlops-heart-disease
```

### View test coverage
```bash
cd Part5
pytest --cov --cov-report=term
open htmlcov/index.html  # or start htmlcov/index.html on Windows
```

---

## 📝 Common File Paths

### Configuration Files
```
requirements.txt                                  # Python dependencies
.github/workflows/ci-cd.yml                      # CI/CD pipeline
Part5/pytest.ini                                 # Pytest configuration
Part6/Dockerfile                                 # Docker image definition
Part6/docker-compose.yml                         # Docker Compose config
Part7/k8s/deployment.yaml                        # Kubernetes deployment
Part8/config/prometheus.yml                      # Prometheus config
Part8/docker-compose-monitoring.yml              # Monitoring stack
```

### Documentation
```
README.md                                        # Main README
COMPLETION_SUMMARY.md                            # Completion summary
QUICK_REFERENCE.md                               # This file
Part9/docs/FINAL_REPORT.md                       # Final report (10+ pages)
Part9/docs/SETUP_INSTRUCTIONS.md                 # Setup guide
Part9/demo/DEMO_SCRIPT.md                        # Demo script
Part9/screenshots/SCREENSHOT_GUIDE.md            # Screenshot guide
```

---

## 🎯 Key Metrics

### Model Performance
- **ROC-AUC**: 91.84%
- **Accuracy**: 82.84%
- **Precision**: 83.42%
- **Recall**: 78.39%

### API Performance
- **Mean Latency**: 45ms
- **p95 Latency**: 80ms
- **Throughput**: ~200 req/s (3 pods)

### Infrastructure
- **Replicas**: 3 (default)
- **Auto-scaling**: 2-10 pods
- **CPU Request**: 250m
- **Memory Request**: 256Mi

---

## 🐛 Common Issues & Solutions

### Issue: Model not found
```bash
python Part4/src/package_model.py
```

### Issue: Port 8000 in use
```bash
# Use different port
docker run -p 8080:8000 heart-disease-api:latest
# Then access: http://localhost:8080
```

### Issue: Tests failing
```bash
# Ensure data is preprocessed
python Part1/src/data_preprocess.py
cd Part5 && pytest -v
```

### Issue: Docker build fails
```bash
# Clean Docker cache
docker system prune -a
# Rebuild
docker build --no-cache -t heart-disease-api:latest -f Part6/Dockerfile .
```

### Issue: Kubernetes pods not starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n mlops-heart-disease
# For Minikube, load image
minikube image load heart-disease-api:latest
```

---

## 📚 Documentation Quick Links

- [Final Report](Part9/docs/FINAL_REPORT.md)
- [Setup Instructions](Part9/docs/SETUP_INSTRUCTIONS.md)
- [Demo Script](Part9/demo/DEMO_SCRIPT.md)
- [Screenshot Guide](Part9/screenshots/SCREENSHOT_GUIDE.md)
- [Completion Summary](COMPLETION_SUMMARY.md)

---

## 🎬 Demo Flow (10 minutes)

1. **Introduction** (1 min) - Project overview
2. **Data & Training** (2 min) - Show data pipeline and model metrics
3. **MLflow** (1 min) - Show experiment tracking
4. **Testing** (1 min) - Run pytest
5. **API** (2 min) - Show Swagger docs and make prediction
6. **Monitoring** (1 min) - Show Prometheus/Grafana
7. **Kubernetes** (1 min) - Show deployment
8. **Wrap-up** (1 min) - Key achievements

---

## ✅ Pre-Submission Checklist

- [ ] All parts 1-9 complete
- [ ] All tests passing
- [ ] Docker image builds
- [ ] API functional
- [ ] Documentation complete
- [ ] README files updated
- [ ] No hardcoded secrets
- [ ] .gitignore configured
- [ ] Screenshots captured
- [ ] Demo script prepared

---

## 📞 Getting Help

1. Check [Setup Instructions](Part9/docs/SETUP_INSTRUCTIONS.md)
2. Review [Final Report](Part9/docs/FINAL_REPORT.md)
3. Consult individual part READMEs
4. Check troubleshooting sections

---

**Last Updated**: December 22, 2025  
**Status**: ✅ Complete and Ready for Submission

