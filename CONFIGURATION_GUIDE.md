# Configuration Guide
## Setting Up Your MLOps Project

This guide will help you configure all necessary values and run the complete project.

---

## Step 1: GitHub Repository Setup

### 1.1 Create GitHub Repository

1. Go to https://github.com
2. Click "New repository"
3. Name it: `MLOP-Assign` or `heart-disease-mlops`
4. Make it **Public** (for GitHub Actions free tier)
5. Don't initialize with README (we already have one)
6. Click "Create repository"

### 1.2 Push Your Code to GitHub

```bash
# Navigate to your project directory
cd "<path-to-your-project-directory>"
# Example: cd "C:\Projects\MLOP-Assign"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Complete MLOps pipeline for heart disease prediction"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/MLOP-Assign.git

# Push to GitHub
git push -u origin main
```

**If you get an error about 'master' vs 'main':**
```bash
git branch -M main
git push -u origin main
```

### 1.3 Update Repository URL in Documentation

**Replace `<repository-url>` in these files:**

1. **README.md** (line 49):
   ```bash
   git clone https://github.com/YOUR_USERNAME/MLOP-Assign.git
   ```

2. **Part9/docs/SETUP_INSTRUCTIONS.md**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MLOP-Assign.git
   ```

3. **Part9/docs/FINAL_REPORT.md** (at the end):
   ```
   **Project Repository**: https://github.com/YOUR_USERNAME/MLOP-Assign
   ```

---

## Step 2: Docker Configuration

### 2.1 No Changes Needed!

The Docker configuration is already complete. However, you should know:

**Docker Hub (Optional - for sharing images):**

If you want to push your Docker image to Docker Hub:

```bash
# Login to Docker Hub
docker login

# Tag your image
docker tag heart-disease-api:latest YOUR_DOCKERHUB_USERNAME/heart-disease-api:latest

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/heart-disease-api:latest
```

**Update in Kubernetes manifests (if using Docker Hub):**

In `Part7/k8s/deployment.yaml`, change:
```yaml
image: heart-disease-api:latest
```
to:
```yaml
image: YOUR_DOCKERHUB_USERNAME/heart-disease-api:latest
```

---

## Step 3: Configuration Values Summary

### Values to Replace

| File | Line/Section | Replace | With |
|------|--------------|---------|------|
| README.md | Line 49 | `<repository-url>` | Your GitHub repo URL |
| Part9/docs/SETUP_INSTRUCTIONS.md | Line 15 | `<repository-url>` | Your GitHub repo URL |
| Part9/docs/FINAL_REPORT.md | End of file | `[GitHub Link]` | Your GitHub repo URL |
| Part9/docs/FINAL_REPORT.md | End of file | `[Your Email]` | Your email address |

### Optional Docker Hub Configuration

| File | Section | Current | Replace With |
|------|---------|---------|--------------|
| Part7/k8s/deployment.yaml | image | `heart-disease-api:latest` | `YOUR_DOCKERHUB_USERNAME/heart-disease-api:latest` |
| Part7/helm/values.yaml | image.repository | `heart-disease-api` | `YOUR_DOCKERHUB_USERNAME/heart-disease-api` |

---

## Step 4: How to Run the Complete Project

### Option A: Complete Fresh Setup (Recommended for First Time)

```bash
# 1. Navigate to project directory
cd "<path-to-your-project-directory>"
# Example: cd "C:\Projects\MLOP-Assign"

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run data pipeline
python Part1/dataset_download_script/download_data.py
python Part1/src/data_preprocess.py
python Part1/src/eda.py

# 6. Train models
python Part2/src/train_models.py

# 7. Track with MLflow
python Part3/src/train_with_mlflow.py

# 8. Package final model
python Part4/src/package_model.py

# 9. Run tests
cd Part5
pytest -v
cd ..

# 10. View MLflow UI (optional - in new terminal)
mlflow ui --backend-store-uri file://$(pwd)/Part3/mlruns
# Access at: http://localhost:5000
```

**Expected Duration**: 10-15 minutes

---

### Option B: Run API Locally (After Setup)

```bash
# Navigate to project
cd "<path-to-your-project-directory>"
# Example: cd "C:\Projects\MLOP-Assign"

# Activate virtual environment
venv\Scripts\activate

# Run API
cd Part6/src
python app.py

# Access API at:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
```

**Test the API:**
```bash
# In another terminal
curl http://localhost:8000/health

# Or open browser and go to:
# http://localhost:8000/docs
```

---

### Option C: Run with Docker

```bash
# 1. Build Docker image
cd "<path-to-your-project-directory>"
# Example: cd "C:\Projects\MLOP-Assign"
docker build -t heart-disease-api:latest -f Part6/Dockerfile .

# 2. Run container
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest

# 3. Check if running
docker ps

# 4. View logs
docker logs -f heart-disease-api

# 5. Test API
curl http://localhost:8000/health

# 6. Stop and remove (when done)
docker stop heart-disease-api
docker rm heart-disease-api
```

---

### Option D: Run with Docker Compose (with Monitoring)

```bash
# Navigate to Part8
cd "<path-to-your-project-directory>\Part8"
# Example: cd "C:\Projects\MLOP-Assign\Part8"

# Start all services
docker-compose -f docker-compose-monitoring.yml up -d

# Access services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)

# View logs
docker-compose -f docker-compose-monitoring.yml logs -f

# Stop all services
docker-compose -f docker-compose-monitoring.yml down
```

---

### Option E: Deploy to Kubernetes (Advanced)

**Prerequisites:**
- Minikube or Docker Desktop with Kubernetes enabled

**For Minikube:**
```bash
# 1. Start Minikube
minikube start --cpus=4 --memory=8192
minikube addons enable metrics-server

# 2. Build and load image
cd "<path-to-your-project-directory>"
# Example: cd "C:\Projects\MLOP-Assign"
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
minikube image load heart-disease-api:latest

# 3. Deploy
cd Part7
# On Windows, run commands from deploy.sh manually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml -n mlops-heart-disease
kubectl apply -f k8s/deployment.yaml -n mlops-heart-disease
kubectl apply -f k8s/hpa.yaml -n mlops-heart-disease

# 4. Check status
kubectl get pods -n mlops-heart-disease
kubectl get services -n mlops-heart-disease

# 5. Access API
minikube service heart-disease-api-service -n mlops-heart-disease --url

# 6. Cleanup (when done)
kubectl delete namespace mlops-heart-disease
minikube stop
```

---

## Step 5: Verify Everything Works

### Quick Verification Checklist

```bash
# 1. Check if data exists
dir Part1\data\interim\heart_clean.csv

# 2. Check if models exist
dir Part4\models\final_model.joblib

# 3. Run tests
cd Part5
pytest -v
cd ..

# 4. Build Docker image
docker build -t heart-disease-api:latest -f Part6/Dockerfile .

# 5. Run container
docker run -d --name test-api -p 8000:8000 heart-disease-api:latest

# 6. Test API
curl http://localhost:8000/health

# 7. Cleanup
docker stop test-api
docker rm test-api
```

**All checks should pass!**

---

## Step 6: Capture Screenshots

Follow the guide in `Part9/screenshots/SCREENSHOT_GUIDE.md` to capture all 54 required screenshots.

**Quick screenshot checklist:**
- [ ] EDA visualizations (5 screenshots)
- [ ] Model training outputs (5 screenshots)
- [ ] MLflow UI (5 screenshots)
- [ ] Test execution (3 screenshots)
- [ ] GitHub Actions (4 screenshots)
- [ ] API endpoints (7 screenshots)
- [ ] Docker (5 screenshots)
- [ ] Kubernetes (6 screenshots)
- [ ] Monitoring (10 screenshots)
- [ ] Additional (4 screenshots)

---

## Step 7: Prepare for Demo

1. **Review Demo Script**: `Part9/demo/DEMO_SCRIPT.md`
2. **Practice**: Run through the demo once
3. **Time yourself**: Should be 10-15 minutes
4. **Prepare talking points**: Know what each component does

---

## Common Issues and Solutions

### Issue 1: Python packages not installing
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt
```

### Issue 2: Docker build fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t heart-disease-api:latest -f Part6/Dockerfile .
```

### Issue 3: Port 8000 already in use
```bash
# Find what's using the port
# Windows:
netstat -ano | findstr :8000

# Kill the process or use different port
docker run -d -p 8080:8000 heart-disease-api:latest
```

### Issue 4: Model not found
```bash
# Ensure Part 4 is completed
python Part4/src/package_model.py
```

### Issue 5: Tests failing
```bash
# Ensure data is preprocessed
python Part1/src/data_preprocess.py

# Run tests with verbose output
cd Part5
pytest -v -s
```

---

## Pre-Submission Checklist

- [ ] GitHub repository created and code pushed
- [ ] Repository URL updated in documentation
- [ ] All parts 1-9 complete
- [ ] All tests passing (`cd Part5 && pytest -v`)
- [ ] Docker image builds successfully
- [ ] API runs and responds to requests
- [ ] MLflow UI accessible
- [ ] Screenshots captured (54 total)
- [ ] Demo script reviewed
- [ ] Final report reviewed
- [ ] No hardcoded secrets or personal information
- [ ] .gitignore configured properly

---

## Quick Command Reference

### Essential Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline
python Part1/dataset_download_script/download_data.py
python Part1/src/data_preprocess.py
python Part2/src/train_models.py
python Part4/src/package_model.py

# Test
cd Part5 && pytest -v && cd ..

# Docker
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
curl http://localhost:8000/health

# MLflow
mlflow ui --backend-store-uri file://$(pwd)/Part3/mlruns
```

---

## Need Help?

1. Check `QUICK_REFERENCE.md` for quick commands
2. Review `Part9/docs/SETUP_INSTRUCTIONS.md` for detailed setup
3. Consult `Part9/docs/FINAL_REPORT.md` for comprehensive documentation
4. Check individual part READMEs for specific issues

---

## You're Ready!

Once you've completed all steps above:
1. Code is on GitHub
2. All parts work locally
3. Docker image builds and runs
4. Tests pass
5. Screenshots captured
6. Demo prepared

**Your project is ready for submission!**

---

**Last Updated**: December 22, 2025  
**Status**: Configuration Guide Complete

