# Complete MLOps Workflow - Parts 1-8

This document provides a bird's-eye view of the entire MLOps workflow and how all parts connect together.

---

## 🎯 Assignment Overview

**Goal:** Build a complete MLOps pipeline from data preprocessing to production deployment with monitoring.

**Dataset:** UCI Heart Disease (Cleveland) - 303 patients, 13 features, binary classification

**Model:** Random Forest Classifier (91.84% ROC-AUC)

---

## 📊 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         PART 1: DATA                            │
│                                                                 │
│  Raw Data → Clean → Encode → EDA Visualizations                │
│  (UCI Cleveland Dataset)                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PART 2: MODEL TRAINING                       │
│                                                                 │
│  Logistic Regression + Random Forest                            │
│  GridSearchCV → Cross-Validation → Best Models                 │
│  Outputs: Models (.joblib) + Metrics (JSON) + Plots (PNG)      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PART 3: MLFLOW TRACKING                       │
│                                                                 │
│  Same training as Part 2 BUT with experiment tracking:         │
│  - Parameters logged                                            │
│  - Metrics logged                                               │
│  - Artifacts saved (models, plots, reports)                     │
│  - Model registry (optional)                                    │
│  View at: http://localhost:5000                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PART 4: MODEL PACKAGING                         │
│                                                                 │
│  Best Model → Production Format:                                │
│  - final_model.joblib (serialized)                              │
│  - mlflow_model/ (MLflow format)                                │
│  - schema.json (input validation)                               │
│  - Inference pipeline tested                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PART 5: TESTING (Optional)                   │
│                                                                 │
│  Unit Tests:                                                    │
│  - Data preprocessing tests                                     │
│  - Feature engineering tests                                    │
│  - Model training tests                                         │
│  - Inference tests                                              │
│  Coverage: pytest --cov                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PART 6: API DEVELOPMENT                      │
│                                                                 │
│  FastAPI REST API:                                              │
│  - GET /health (health check)                                   │
│  - POST /predict (single prediction)                            │
│  - POST /batch_predict (batch predictions)                      │
│  - GET /model/info (model metadata)                             │
│  - GET /docs (Swagger UI)                                       │
│  Dockerized: heart-disease-api:latest                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              PART 7: KUBERNETES DEPLOYMENT                      │
│                                                                 │
│  Minikube Cluster:                                              │
│  - 3 API pod replicas                                           │
│  - LoadBalancer service                                         │
│  - Health probes (liveness + readiness)                         │
│  - Resource limits (CPU + memory)                               │
│  - Auto-scaling ready                                           │
│  Access: minikube service heart-disease-api-service --url       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              PART 8: MONITORING & LOGGING                       │
│                                                                 │
│  Docker Compose Stack:                                          │
│  ┌─────────────────────────────────────────────────┐           │
│  │  API (Port 8000)                                │           │
│  │  - Prometheus metrics at /metrics               │           │
│  │  - Structured logging (JSON)                    │           │
│  └──────────────────┬──────────────────────────────┘           │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────────────┐           │
│  │  Prometheus (Port 9090)                         │           │
│  │  - Scrapes /metrics every 10s                   │           │
│  │  - Stores time-series data                      │           │
│  │  - PromQL queries                               │           │
│  └──────────────────┬──────────────────────────────┘           │
│                     │                                           │
│                     ↓                                           │
│  ┌─────────────────────────────────────────────────┐           │
│  │  Grafana (Port 3000)                            │           │
│  │  - Visualizes metrics                           │           │
│  │  - Custom dashboards                            │           │
│  │  - Real-time monitoring                         │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Training Phase (Parts 1-4)
```
Raw Data → Preprocessing → Feature Engineering → Model Training → 
Best Model Selection → Model Packaging → Production Model
```

### Inference Phase (Parts 6-8)
```
API Request → Input Validation → Model Prediction → 
Response + Logging + Metrics → Client
```

### Monitoring Phase (Part 8)
```
API Metrics → Prometheus Collection → Grafana Visualization → 
Insights & Alerts
```

---

## 📁 Key Files Generated

### Part 1 Outputs
```
Part1/
├── data/
│   ├── raw/processed.cleveland.data          # Original dataset
│   ├── interim/heart_clean.csv                # Cleaned data
│   └── processed/
│       ├── heart_encoded.csv                  # Encoded features
│       └── feature_names.json                 # Feature list
└── reports/figures/
    ├── histograms_numeric.png                 # Feature distributions
    ├── corr_heatmap.png                       # Correlation matrix
    └── class_balance.png                      # Target distribution
```

### Part 2 Outputs
```
Part2/outputs/
├── models/
│   ├── logreg_best.joblib                     # Logistic Regression
│   └── rf_best.joblib                         # Random Forest
├── metrics/
│   ├── logreg_cv_metrics.json                 # LR metrics
│   ├── rf_cv_metrics.json                     # RF metrics
│   └── scores_summary.csv                     # Comparison
└── plots/
    ├── logreg_confusion_matrix.png
    ├── logreg_roc_curve.png
    ├── rf_confusion_matrix.png
    └── rf_roc_curve.png
```

### Part 3 Outputs
```
Part3/
├── mlruns/                                    # MLflow tracking
│   └── <experiment_id>/
│       ├── <run_id>/                          # Each run
│       │   ├── params/                        # Hyperparameters
│       │   ├── metrics/                       # Performance metrics
│       │   └── artifacts/                     # Models, plots
│       └── models/                            # Registered models
└── outputs/                                   # Same as Part2
```

### Part 4 Outputs
```
Part4/
├── models/
│   ├── final_model.joblib                     # Production model
│   ├── mlflow_model/                          # MLflow format
│   │   ├── MLmodel
│   │   ├── model.pkl
│   │   ├── conda.yaml
│   │   └── requirements.txt
│   └── schema.json                            # Input schema
├── metrics/
│   └── final_report.json                      # Model metadata
└── outputs/
    └── predictions.csv                        # Test predictions
```

### Part 6 Outputs
```
Part6/
├── Dockerfile                                 # Container definition
├── docker-compose.yml                         # Local deployment
└── src/app.py                                 # FastAPI application
```

### Part 7 Outputs
```
Part7/k8s/
├── deployment.yaml                            # K8s deployment
├── service.yaml                               # LoadBalancer
├── configmap.yaml                             # Configuration
├── hpa.yaml                                   # Auto-scaling
└── ingress.yaml                               # Ingress rules
```

### Part 8 Outputs
```
Part8/
├── docker-compose-monitoring.yml              # Full stack
├── config/
│   ├── prometheus.yml                         # Prometheus config
│   └── grafana-datasource.yml                 # Grafana datasource
├── dashboards/
│   └── api-dashboard.json                     # Grafana dashboard
├── logs/
│   └── api.log                                # Application logs
└── src/
    └── app_with_monitoring.py                 # Instrumented API
```

---

## 🎯 Learning Objectives Achieved

### Part 1: Data Engineering
✅ Data acquisition and cleaning  
✅ Exploratory data analysis  
✅ Feature encoding and preprocessing  
✅ Data validation

### Part 2: Machine Learning
✅ Model selection and training  
✅ Hyperparameter tuning (GridSearchCV)  
✅ Cross-validation  
✅ Model evaluation and comparison

### Part 3: Experiment Tracking
✅ MLflow setup and configuration  
✅ Parameter and metric logging  
✅ Artifact management  
✅ Model registry

### Part 4: Model Packaging
✅ Model serialization  
✅ MLflow model format  
✅ Schema definition  
✅ Inference pipeline

### Part 5: Testing (Optional)
✅ Unit testing  
✅ Integration testing  
✅ Code coverage  
✅ Test automation

### Part 6: API Development
✅ REST API design  
✅ FastAPI implementation  
✅ API documentation (Swagger)  
✅ Docker containerization

### Part 7: Orchestration
✅ Kubernetes deployment  
✅ Service exposure  
✅ Health checks  
✅ Scaling configuration

### Part 8: Monitoring
✅ Metrics collection (Prometheus)  
✅ Visualization (Grafana)  
✅ Structured logging  
✅ Observability

---

## 🔧 Technologies Used

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Data** | Pandas, NumPy | Data manipulation |
| **Visualization** | Matplotlib, Seaborn | EDA and plots |
| **ML** | Scikit-learn | Model training |
| **Tracking** | MLflow | Experiment management |
| **API** | FastAPI, Uvicorn | REST API |
| **Validation** | Pydantic | Input validation |
| **Testing** | Pytest | Unit tests |
| **Containerization** | Docker | Packaging |
| **Orchestration** | Kubernetes, Minikube | Deployment |
| **Monitoring** | Prometheus | Metrics collection |
| **Visualization** | Grafana | Dashboards |
| **Logging** | Python logging | Application logs |

---

## 📊 Metrics Tracked

### Model Metrics (Parts 2-4)
- Accuracy
- Precision
- Recall
- ROC-AUC
- F1-Score
- Confusion Matrix

### API Metrics (Part 8)
- `api_requests_total` - Total requests by endpoint/method/status
- `api_request_duration_seconds` - Request latency
- `predictions_total` - Predictions by class and risk level
- `prediction_duration_seconds` - Prediction processing time
- `active_requests` - Current active requests
- `api_errors_total` - Errors by type and endpoint
- `model_load_time_seconds` - Model initialization time

### System Metrics (Part 8)
- CPU usage
- Memory usage
- Network I/O
- Disk I/O

---

## 🚀 Deployment Options

### Development
```powershell
# Local API
python Part6/src/app.py
```

### Testing
```powershell
# Docker container
docker run -p 8000:8000 heart-disease-api:latest
```

### Production (Option 1: Kubernetes)
```powershell
# Minikube cluster
kubectl apply -f Part7/k8s/deployment.yaml
minikube tunnel
```

### Production (Option 2: Monitoring Stack)
```powershell
# Docker Compose with monitoring
cd Part8
docker-compose -f docker-compose-monitoring.yml up -d
```

---

## 📈 Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Model ROC-AUC** | 91.84% | Random Forest (best) |
| **API Latency (p50)** | ~45ms | Single prediction |
| **API Latency (p95)** | ~80ms | Single prediction |
| **API Latency (p99)** | ~120ms | Single prediction |
| **Throughput** | ~20 req/s | Single container |
| **Model Load Time** | ~1.3s | Startup time |
| **Container Size** | ~800MB | Optimized multi-stage |
| **Memory Usage** | ~250MB | Per container |
| **CPU Usage** | ~0.2 cores | Idle state |

---

## 🎓 Best Practices Demonstrated

### Code Quality
✅ Modular code structure  
✅ Type hints and docstrings  
✅ Error handling  
✅ Logging throughout

### ML Engineering
✅ Reproducible experiments  
✅ Model versioning  
✅ Feature validation  
✅ Cross-validation

### DevOps
✅ Containerization  
✅ Infrastructure as Code  
✅ Health checks  
✅ Resource limits

### Monitoring
✅ Comprehensive metrics  
✅ Structured logging  
✅ Dashboards  
✅ Alerting ready

---

## 📝 Assignment Deliverables

### Code & Outputs
- [ ] All Python scripts (Parts 1-4)
- [ ] Generated datasets and features
- [ ] Trained models (.joblib files)
- [ ] MLflow experiment tracking
- [ ] FastAPI application
- [ ] Dockerfile and docker-compose files
- [ ] Kubernetes manifests

### Documentation
- [ ] README files for each part
- [ ] Code comments and docstrings
- [ ] API documentation (Swagger)
- [ ] Deployment guides

### Screenshots (30 total)
- [ ] Part 7: Kubernetes deployment (12 screenshots)
- [ ] Part 8: Monitoring stack (18 screenshots)
- See `Part8/SCREENSHOT_GUIDE.md` for details

### Metrics & Reports
- [ ] Model performance metrics (JSON)
- [ ] MLflow experiment results
- [ ] API test results
- [ ] Monitoring dashboards

---

## 🔗 Quick Navigation

### Getting Started
→ [QUICK_START.md](../QUICK_START.md) - Start here!

### Complete Guide
→ [COMPLETE_SETUP_GUIDE.md](../COMPLETE_SETUP_GUIDE.md) - Full instructions

### Automation
→ [run_all_parts.ps1](../run_all_parts.ps1) - Automated setup

### Deployment
→ [DEPLOYMENT_STEPS.md](DEPLOYMENT_STEPS.md) - Steps 7 & 8

### Testing
→ [POWERSHELL_COMMANDS.md](POWERSHELL_COMMANDS.md) - Command reference

### Screenshots
→ [SCREENSHOT_GUIDE.md](SCREENSHOT_GUIDE.md) - Documentation guide

---

## ✅ Success Criteria

Your assignment is complete when:

1. ✅ All parts (1-8) run successfully
2. ✅ Models achieve >85% ROC-AUC
3. ✅ API responds to all endpoints
4. ✅ Kubernetes deployment is accessible
5. ✅ Monitoring dashboard shows metrics
6. ✅ All screenshots captured
7. ✅ Documentation is complete

---

## 🎉 Congratulations!

You've built a complete MLOps pipeline covering:
- Data engineering
- Machine learning
- Experiment tracking
- API development
- Containerization
- Orchestration
- Monitoring & logging

This is production-grade ML engineering! 🚀

---

**For questions or issues, refer to the troubleshooting sections in each guide.**

