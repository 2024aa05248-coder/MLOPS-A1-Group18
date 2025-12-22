# Demo Script
## Heart Disease Prediction MLOps Project

This script provides a step-by-step demonstration of the complete MLOps pipeline.

**Duration**: 10-15 minutes  
**Audience**: Technical stakeholders, instructors, team members

---

## Demo Preparation

### Before Starting
1. ✅ Complete setup (all parts 1-4 trained and packaged)
2. ✅ Docker installed and running
3. ✅ Terminal/command prompt ready
4. ✅ Browser open for API docs and monitoring

### Quick Setup Check
```bash
# Verify model exists
ls Part4/models/final_model.joblib

# Verify Docker is running
docker --version
docker ps
```

---

## Demo Flow

### Part 1: Introduction (1 minute)

**Script**:
> "Today I'll demonstrate a complete MLOps pipeline for heart disease prediction. This project showcases:
> - End-to-end ML workflow from data to deployment
> - Automated CI/CD pipeline
> - Production-ready API with monitoring
> - Kubernetes orchestration
> 
> Let's start with the data and model training."

---

### Part 2: Data & Model Training (2 minutes)

#### Show Data Pipeline

```bash
# Show raw data
head -5 Part1/data/raw/processed.cleveland.data

# Show cleaned data
head -5 Part1/data/interim/heart_clean.csv
```

**Script**:
> "We start with the UCI Heart Disease dataset - 303 patients with 14 health features. 
> The pipeline automatically downloads, cleans, and preprocesses the data."

#### Show EDA Results

```bash
# Open EDA plots
# Windows: start Part1/reports/figures/corr_heatmap.png
# macOS: open Part1/reports/figures/corr_heatmap.png
# Linux: xdg-open Part1/reports/figures/corr_heatmap.png
```

**Script**:
> "Our exploratory analysis reveals key correlations. Features like exercise-induced angina 
> and ST depression show strong predictive power."

#### Show Model Performance

```bash
# Display model metrics
cat Part2/outputs/metrics/scores_summary.csv
```

**Script**:
> "We trained two models - Logistic Regression and Random Forest. Both achieve over 91% ROC-AUC.
> Random Forest was selected for production with 91.84% ROC-AUC and lower variance."

---

### Part 3: Experiment Tracking (2 minutes)

#### Launch MLflow UI

```bash
mlflow ui --backend-store-uri file://$(pwd)/Part3/mlruns &
```

**Open Browser**: http://localhost:5000

**Script**:
> "All experiments are tracked with MLflow. Here you can see:
> - Multiple training runs with different hyperparameters
> - Metrics comparison across models
> - Logged artifacts like plots and confusion matrices
> - Model registry with versioning
> 
> This ensures complete reproducibility - anyone can recreate our results."

**Demo Actions**:
1. Show experiment list
2. Click on a run to show details
3. Show metrics comparison chart
4. Show logged artifacts (ROC curve, confusion matrix)

---

### Part 4: Testing & CI/CD (2 minutes)

#### Run Tests

```bash
cd Part5
pytest -v --tb=short
cd ..
```

**Script**:
> "We have comprehensive test coverage with 40 unit tests covering:
> - Data preprocessing
> - Feature engineering
> - Model training and evaluation
> - Inference pipeline
> 
> All tests are automated in our CI/CD pipeline."

#### Show CI/CD Pipeline

**Open Browser**: GitHub repository → Actions tab

**Script**:
> "Our GitHub Actions pipeline automatically:
> 1. Runs linting and code quality checks
> 2. Executes all unit tests
> 3. Trains models and logs to MLflow
> 4. Uploads artifacts
> 
> Every push triggers this pipeline, ensuring code quality and model performance."

---

### Part 5: API Deployment (3 minutes)

#### Build and Run Docker Container

```bash
# Build image (if not already built)
docker build -t heart-disease-api:latest -f Part6/Dockerfile .

# Run container
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest

# Wait a few seconds for startup
sleep 5

# Check health
curl http://localhost:8000/health
```

**Script**:
> "The model is packaged as a Docker container running a FastAPI application.
> This ensures consistency across environments - it works the same on my laptop, 
> your server, or in the cloud."

#### Show API Documentation

**Open Browser**: http://localhost:8000/docs

**Script**:
> "The API provides interactive documentation with Swagger UI. Let's test it live."

**Demo Actions**:
1. Show available endpoints
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Use the example patient data
5. Execute request
6. Show response with prediction, probability, and risk level

#### Test Prediction via curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

**Script**:
> "The API returns:
> - Prediction: 0 (no disease) or 1 (disease present)
> - Probability: Confidence score for the prediction
> - Risk Level: Low, Medium, or High
> - Timestamp: For audit trail
> 
> Response time is typically under 50 milliseconds."

#### Show Model Info

```bash
curl http://localhost:8000/model/info
```

**Script**:
> "We can query the API for model metadata including performance metrics."

---

### Part 6: Monitoring (2 minutes)

#### Start Monitoring Stack

```bash
cd Part8
docker-compose -f docker-compose-monitoring.yml up -d
sleep 10
cd ..
```

#### Show Prometheus Metrics

**Open Browser**: http://localhost:8000/metrics

**Script**:
> "The API exposes Prometheus metrics including:
> - Request counts and rates
> - Response times (histograms)
> - Prediction distribution by risk level
> - Error counts
> - Active requests"

#### Show Grafana Dashboard

**Open Browser**: http://localhost:3000 (admin/admin)

**Script**:
> "Grafana provides real-time visualization of our metrics:
> - Request rate over time
> - Latency percentiles (p50, p95, p99)
> - Prediction distribution
> - Error rates
> 
> This enables proactive monitoring and quick issue detection."

**Demo Actions**:
1. Login to Grafana
2. Navigate to dashboards
3. Show API monitoring dashboard
4. Point out key panels

#### Generate Some Load

```bash
# Run test script to generate traffic
cd Part6/src
python test_api.py
cd ../..
```

**Script**:
> "Let's generate some traffic and watch the metrics update in real-time."

**Refresh Grafana dashboard to show updated metrics**

---

### Part 7: Kubernetes Deployment (2 minutes)

**Note**: This section is optional if Kubernetes is not set up

#### Show Kubernetes Manifests

```bash
ls Part7/k8s/
```

**Script**:
> "For production deployment, we use Kubernetes with:
> - Deployment: 3 replicas for high availability
> - Service: LoadBalancer for external access
> - HPA: Auto-scaling from 2-10 pods based on CPU/memory
> - ConfigMap: Configuration management
> - Ingress: Routing and SSL termination"

#### Show Deployment (if running)

```bash
kubectl get pods -n mlops-heart-disease
kubectl get services -n mlops-heart-disease
kubectl get hpa -n mlops-heart-disease
```

**Script**:
> "Kubernetes provides:
> - Automatic scaling based on load
> - Self-healing (restarts failed pods)
> - Rolling updates with zero downtime
> - Resource management and isolation"

---

### Part 8: Architecture Overview (1 minute)

**Show Architecture Diagram** (from Final Report)

**Script**:
> "Let's review the complete architecture:
> 
> 1. **Data Layer**: Automated data pipeline with preprocessing
> 2. **ML Layer**: Model training with experiment tracking
> 3. **API Layer**: FastAPI serving predictions
> 4. **Container Layer**: Docker for consistency
> 5. **Orchestration Layer**: Kubernetes for scalability
> 6. **Monitoring Layer**: Prometheus and Grafana for observability
> 7. **CI/CD Layer**: GitHub Actions for automation
> 
> This is a production-ready MLOps system following industry best practices."

---

### Part 9: Key Highlights (1 minute)

**Script**:
> "Key achievements of this project:
> 
> ✅ **High Performance**: 91.84% ROC-AUC on heart disease prediction
> ✅ **Fully Automated**: CI/CD pipeline from code to deployment
> ✅ **Production Ready**: Containerized, scalable, monitored
> ✅ **Reproducible**: Complete experiment tracking and versioning
> ✅ **Well Tested**: 40 unit tests with good coverage
> ✅ **Observable**: Comprehensive logging and metrics
> ✅ **Scalable**: Auto-scaling from 2-10 pods based on load
> ✅ **Documented**: Extensive documentation and runbooks
> 
> This system can handle real-world production traffic and serves as a 
> template for other ML projects."

---

### Part 10: Q&A and Cleanup (2 minutes)

**Script**:
> "I'm happy to answer any questions about:
> - Model performance and selection
> - MLOps pipeline and automation
> - Deployment and scaling
> - Monitoring and observability
> - Future improvements"

#### Cleanup After Demo

```bash
# Stop Docker containers
docker stop heart-disease-api
docker rm heart-disease-api

# Stop monitoring stack
cd Part8
docker-compose -f docker-compose-monitoring.yml down
cd ..

# Stop MLflow UI (if running in background)
pkill -f "mlflow ui"

# Stop Kubernetes (if running)
# cd Part7
# ./undeploy.sh
# minikube stop
```

---

## Demo Tips

### Preparation
- ✅ Run through the demo once before presenting
- ✅ Have all terminals and browsers pre-positioned
- ✅ Ensure stable internet connection
- ✅ Have backup screenshots in case of issues
- ✅ Time yourself to stay within limits

### During Demo
- 🎯 Focus on key features, don't get lost in details
- 🎯 Explain WHY not just WHAT (e.g., why MLflow, why Kubernetes)
- 🎯 Show real outputs, not just code
- 🎯 Highlight production-readiness aspects
- 🎯 Connect technical features to business value

### Handling Issues
- 🔧 If something fails, have screenshots ready
- 🔧 Explain what SHOULD happen
- 🔧 Move on quickly, don't debug live
- 🔧 Acknowledge issues professionally

### Time Management
- ⏱️ Introduction: 1 min
- ⏱️ Data & Training: 2 min
- ⏱️ Experiment Tracking: 2 min
- ⏱️ Testing & CI/CD: 2 min
- ⏱️ API Deployment: 3 min
- ⏱️ Monitoring: 2 min
- ⏱️ Kubernetes: 2 min (optional)
- ⏱️ Architecture: 1 min
- ⏱️ Wrap-up: 1 min
- ⏱️ Buffer: 2 min

**Total**: 15 minutes (18 with Kubernetes)

---

## Alternative: Quick Demo (5 minutes)

For a shorter demo, focus on:

1. **Show Final Model Performance** (1 min)
   - Display metrics from Part2/outputs/metrics/
   
2. **API Demo** (2 min)
   - Run Docker container
   - Show Swagger docs
   - Make a prediction

3. **Monitoring** (1 min)
   - Show Prometheus metrics endpoint
   - Show one Grafana dashboard

4. **Architecture Overview** (1 min)
   - Show architecture diagram
   - Highlight key components

---

## Video Recording Tips

If recording a demo video:

1. **Screen Recording**:
   - Use OBS Studio or similar
   - Record at 1080p minimum
   - Enable audio for narration

2. **Editing**:
   - Speed up slow parts (docker build, model training)
   - Add captions for key points
   - Include title cards for each section

3. **Structure**:
   - Introduction (30s)
   - Main demo (8-10 min)
   - Conclusion (30s)
   - Total: 10-12 minutes

---

**Good luck with your demo!**

