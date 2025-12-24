# Complete Deployment Guide - Steps 7 & 8

This guide provides step-by-step instructions to deploy the Heart Disease Prediction API using Docker and Kubernetes (Minikube), followed by setting up comprehensive monitoring with Prometheus and Grafana.

---

## Prerequisites

Ensure you have the following installed:
- **Docker Desktop** (with Kubernetes enabled)
- **Minikube**
- **kubectl**

---

## STEP 7: Deploy to Minikube (Kubernetes)

### 7.1 Build the Docker Image

Navigate to the project root and build the Docker image:

```powershell
cd "C:\Users\ashmitad\Documents\Personal\Bits\SEMESTER 3\MLOPs\Assignment 1\MLOP-Assign"
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
```

**Expected Output:** Image builds successfully with all layers completed.

### 7.2 Start Minikube

```powershell
minikube start --driver=docker
```

**Wait Time:** 2-3 minutes for Minikube to fully start.

### 7.3 Load Docker Image into Minikube

Since Minikube runs in its own Docker environment, load your image:

```powershell
minikube image load heart-disease-api:latest
```

Verify the image is loaded:

```powershell
minikube image ls | findstr heart-disease-api
```

### 7.4 Create Kubernetes Deployment

Apply the deployment manifest:

```powershell
kubectl apply -f Part7/k8s/deployment.yaml
```

**Expected Output:**
```
deployment.apps/heart-disease-api created
service/heart-disease-api-service created
```

### 7.5 Verify Deployment

Check if pods are running:

```powershell
kubectl get pods
```

**Expected Output (after 1-2 minutes):**
```
NAME                                  READY   STATUS    RESTARTS   AGE
heart-disease-api-xxxxxxxxxx-xxxxx    1/1     Running   0          30s
heart-disease-api-xxxxxxxxxx-xxxxx    1/1     Running   0          30s
heart-disease-api-xxxxxxxxxx-xxxxx    1/1     Running   0          30s
```

Check deployment status:

```powershell
kubectl get deployment heart-disease-api
```

Check service:

```powershell
kubectl get service heart-disease-api-service
```

### 7.6 Expose Service via Minikube Tunnel

**Open a NEW terminal window** and run:

```powershell
minikube tunnel
```

**IMPORTANT:** Keep this terminal running - it creates a network route to expose LoadBalancer services.

### 7.7 Get Service URL

In your original terminal:

```powershell
kubectl get service heart-disease-api-service
```

Look for the `EXTERNAL-IP` column. Once it shows an IP (not `<pending>`), you can access the API.

Or use:

```powershell
minikube service heart-disease-api-service --url
```

This will give you the URL (e.g., `http://127.0.0.1:xxxxx`)\
Keeop this terminal open

### 7.8 Test the Deployment

Test the health endpoint:(in new terminal)

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:<PORT>/health"
```

Test the root endpoint:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:<PORT>/"
```

Test a prediction:

```powershell
$body = @{
    age=63; sex=1; cp=1; trestbps=145; chol=233; 
    fbs=1; restecg=2; thalach=150; exang=0; 
    oldpeak=2.3; slope=3; ca=0; thal=6
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:<PORT>/predict" -Method Post -Body $body -ContentType "application/json"
```

### 7.9 Screenshots for Documentation

Take screenshots of:
1. ✅ `kubectl get pods` - All 3 pods in Running state
2. ✅ `kubectl get deployment` - Deployment status
3. ✅ `kubectl get service` - Service with EXTERNAL-IP
4. ✅ Browser at `http://<EXTERNAL-IP>/docs` - API documentation
5. ✅ Successful prediction response in browser/Postman
6. ✅ `kubectl describe deployment heart-disease-api` - Deployment details

---

## STEP 8: Full Monitoring Stack (Docker Compose)

### 8.1 Stop Minikube (Optional - to free resources)

If you want to free up resources:

```powershell
# In the terminal running minikube tunnel, press Ctrl+C
minikube stop
```

Or keep it running if you have enough resources.

### 8.2 Navigate to Part8 Directory

```powershell
cd "C:\Users\ashmitad\Documents\Personal\Bits\SEMESTER 3\MLOPs\Assignment 1\MLOP-Assign\Part8"
```

### 8.3 Start the Full Monitoring Stack

```powershell
docker-compose -f docker-compose-monitoring.yml up -d
```

This starts:
- **API** on port 8000
- **Prometheus** on port 9090
- **Grafana** on port 3000

**Expected Output:**
```
Creating network "monitoring" with driver bridge
Creating volume "prometheus-data"
Creating volume "grafana-data"
Creating heart-disease-api-monitored ... done
Creating prometheus                  ... done
Creating grafana                     ... done
```

### 8.4 Verify All Containers are Running

```powershell
docker-compose -f docker-compose-monitoring.yml ps
```

**Expected Output:**
```
NAME                          IMAGE                    STATUS                    PORTS
grafana                       grafana/grafana:latest   Up                        0.0.0.0:3000->3000/tcp
heart-disease-api-monitored   part8-api                Up (healthy)              0.0.0.0:8000->8000/tcp
prometheus                    prom/prometheus:latest   Up                        0.0.0.0:9090->9090/tcp
```

Check logs if needed:

```powershell
docker-compose -f docker-compose-monitoring.yml logs api
docker-compose -f docker-compose-monitoring.yml logs prometheus
docker-compose -f docker-compose-monitoring.yml logs grafana
```

### 8.5 Access and Test the API

Test the API with monitoring:

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Root endpoint
Invoke-RestMethod -Uri "http://localhost:8000/"

# Metrics endpoint (Prometheus format)
Invoke-WebRequest -Uri "http://localhost:8000/metrics"

# Make a prediction
$body = @{
    age=63; sex=1; cp=1; trestbps=145; chol=233; 
    fbs=1; restecg=2; thalach=150; exang=0; 
    oldpeak=2.3; slope=3; ca=0; thal=6
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

### 8.6 Access Prometheus

1. Open browser: **http://localhost:9090**
2. Go to **Status** → **Targets** to verify API is being scraped
3. Target should show as **UP** with green indicator
4. In the query box, try these queries:
   - `api_requests_total` - Total API requests
   - `api_request_duration_seconds_sum` - Request duration
   - `predictions_total` - Total predictions made
   - `rate(api_requests_total[5m])` - Request rate per second
   - `prediction_duration_seconds_sum` - Total prediction time

### 8.7 Access Grafana

1. Open browser: **http://localhost:3000**
2. Login with:
   - **Username:** `admin`
   - **Password:** `admin`
3. Skip password change or set a new one

### 8.8 Configure Grafana Dashboard

**Verify Prometheus Data Source:**
1. Go to **Connections** (⚙️) → **Data Sources**
2. You should see **Prometheus** already configured (auto-provisioned)
3. Click on it and click **Save & Test** - should show "Data source is working"

**Import Dashboard:**
1. Go to **Dashboards** (☷) → **Browse** → **Import**
2. Click **Upload JSON file**
3. Select `dashboards/api-dashboard.json` from Part8 folder
4. Select **Prometheus** as the data source
5. Click **Import**

**Alternative - Create Custom Dashboard:**
1. Go to **Dashboards** → **New Dashboard**
2. Add panels with queries like:
   - `rate(api_requests_total[5m])` - Request rate
   - `api_request_duration_seconds_sum / api_request_duration_seconds_count` - Avg response time
   - `predictions_total` - Total predictions by risk level
   - `active_requests` - Current active requests

### 8.9 Generate Traffic and View Metrics

Generate some API traffic to populate metrics:

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

### 8.10 View Logs

Check the API logs:

```powershell
# View logs in real-time
docker-compose -f docker-compose-monitoring.yml logs -f api

# Or check the log file directly
Get-Content logs\api.log -Tail 50
```

**Log Format:** Each request is logged with:
- Timestamp
- HTTP method and endpoint
- Client IP address
- Response status code
- Request duration
- Prediction details (for /predict endpoint)

### 8.11 Screenshots for Documentation

Take screenshots of:
1. ✅ `docker-compose ps` - All 3 containers running
2. ✅ API response at `http://localhost:8000/`
3. ✅ API docs at `http://localhost:8000/docs`
4. ✅ API metrics at `http://localhost:8000/metrics`
5. ✅ Prometheus targets page showing API target as "UP"
6. ✅ Prometheus query results for `api_requests_total`
7. ✅ Prometheus graph showing `rate(api_requests_total[5m])`
8. ✅ Grafana login page
9. ✅ Grafana data sources showing Prometheus connected
10. ✅ Grafana dashboard showing API metrics
11. ✅ Terminal showing API logs with request details
12. ✅ Successful prediction responses

---

## Monitoring Metrics Available

### Request Metrics
- `api_requests_total{method, endpoint, status}` - Total requests by method, endpoint, and status
- `api_request_duration_seconds{method, endpoint}` - Request duration histogram
- `active_requests` - Current number of active requests

### Prediction Metrics
- `predictions_total{prediction_class, risk_level}` - Total predictions by class and risk level
- `prediction_duration_seconds` - Prediction processing time histogram

### Model Metrics
- `model_load_time_seconds` - Time taken to load the model

### Error Metrics
- `api_errors_total{error_type, endpoint}` - Total errors by type and endpoint

---

## Cleanup Commands

### Stop Monitoring Stack:
```powershell
cd Part8
docker-compose -f docker-compose-monitoring.yml down
```

### Stop and Clean Monitoring Stack (including volumes):
```powershell
docker-compose -f docker-compose-monitoring.yml down -v
```

### Stop Minikube:
```powershell
kubectl delete -f Part7/k8s/deployment.yaml
minikube stop
```

### Delete Minikube Cluster:
```powershell
minikube delete
```

---

## Quick Verification Checklist

### Step 7 (Minikube) ✅
- [ ] Minikube is running: `minikube status`
- [ ] Image loaded: `minikube image ls | findstr heart-disease-api`
- [ ] Pods running: `kubectl get pods` (all 3 pods in Running state)
- [ ] Service exposed: `kubectl get service` (EXTERNAL-IP assigned)
- [ ] API accessible: `curl http://<EXTERNAL-IP>/health`
- [ ] Prediction works: Test `/predict` endpoint
- [ ] Screenshots taken

### Step 8 (Monitoring) ✅
- [x] All containers running: `docker-compose ps` (3 containers up)
- [x] API accessible: `http://localhost:8000/`
- [x] Metrics endpoint: `http://localhost:8000/metrics`
- [x] Prometheus accessible: `http://localhost:9090`
- [x] Prometheus scraping API: Check Targets page
- [x] Grafana accessible: `http://localhost:3000`
- [x] Grafana dashboard configured
- [x] Logs being generated: Check logs
- [ ] Screenshots taken

---

## Troubleshooting

### Issue: API Container Constantly Restarting

**Solution:** This was fixed by updating the path handling in `app_with_monitoring.py`:
- Changed `PROJECT_ROOT = Path(__file__).resolve().parents[2]` to `parents[0]`
- Updated model paths from `Part4/models/` to `models/`
- Updated metrics paths from `Part4/metrics/` to `metrics/`

### Issue: Minikube Image Load Fails

```powershell
# Use Docker's save/load method
docker save heart-disease-api:latest | minikube image load -
```

### Issue: Pods Not Starting

```powershell
# Check pod logs
kubectl logs <pod-name>

# Describe pod for events
kubectl describe pod <pod-name>
```

### Issue: Docker Compose Fails to Start

```powershell
# Check individual container logs
docker-compose -f docker-compose-monitoring.yml logs <service-name>

# Rebuild if needed
docker-compose -f docker-compose-monitoring.yml build --no-cache
docker-compose -f docker-compose-monitoring.yml up -d
```

### Issue: Prometheus Can't Scrape API

- Verify API container is running: `docker ps`
- Check Prometheus config: `type config\prometheus.yml`
- Verify network: `docker network ls`
- Check API metrics endpoint: `curl http://localhost:8000/metrics`
- Ensure target is `api:8000` (not `localhost:8000`) in prometheus.yml

### Issue: Grafana Can't Connect to Prometheus

- Verify Prometheus is running: `docker ps`
- Check datasource URL is `http://prometheus:9090` (not `localhost`)
- Verify both are on the same Docker network: `monitoring`

---

## Architecture Overview

### Minikube Deployment (Step 7)
```
Internet → LoadBalancer Service → 3 API Pods (Replicas) → ML Model
```

### Monitoring Stack (Step 8)
```
API Container (Port 8000)
    ↓ /metrics endpoint
Prometheus (Port 9090) - Scrapes metrics every 10s
    ↓ PromQL queries
Grafana (Port 3000) - Visualizes metrics
```

---

## Key Files

- `Part6/Dockerfile` - API container definition
- `Part7/k8s/deployment.yaml` - Kubernetes deployment manifest
- `Part8/docker-compose-monitoring.yml` - Full monitoring stack
- `Part8/config/prometheus.yml` - Prometheus configuration
- `Part8/config/grafana-datasource.yml` - Grafana datasource config
- `Part8/src/app_with_monitoring.py` - API with monitoring instrumentation

---

## Success Indicators

### Step 7 Success:
✅ 3 pods running in Kubernetes
✅ LoadBalancer service has external IP
✅ API responds to health checks
✅ Predictions return correct format
✅ All screenshots captured

### Step 8 Success:
✅ All 3 containers healthy
✅ API logs requests with details
✅ Prometheus shows API target as UP
✅ Metrics visible in Prometheus queries
✅ Grafana connects to Prometheus
✅ Dashboard displays real-time metrics
✅ All screenshots captured

---

**Deployment completed successfully! 🎉**

For any issues, check the troubleshooting section or review container logs.

