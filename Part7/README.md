# Part 7 — Production Deployment

## Scope
- Kubernetes deployment manifests for production-ready deployment
- Horizontal Pod Autoscaling (HPA) for automatic scaling
- Service configuration with LoadBalancer
- Ingress configuration for external access
- Helm chart for simplified deployment
- Deployment scripts for automation

## Deployment Options

### Option 1: Local Kubernetes (Minikube/Docker Desktop)
Best for development and testing.

### Option 2: Cloud Kubernetes (GKE/EKS/AKS)
Production-ready deployment on cloud platforms.

## Prerequisites

### For Local Deployment (Minikube)
```bash
# Install Minikube
# Windows: choco install minikube
# Mac: brew install minikube
# Linux: curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64

# Start Minikube
minikube start --cpus=4 --memory=8192

# Enable metrics-server for HPA
minikube addons enable metrics-server

# Verify cluster
kubectl cluster-info
```

### For Docker Desktop Kubernetes
```bash
# Enable Kubernetes in Docker Desktop settings
# Verify
kubectl cluster-info
```

### For Cloud Deployment
```bash
# GKE
gcloud container clusters create heart-disease-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --zone=us-central1-a

# EKS
eksctl create cluster \
  --name heart-disease-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3

# AKS
az aks create \
  --resource-group myResourceGroup \
  --name heart-disease-cluster \
  --node-count 3 \
  --node-vm-size Standard_DS2_v2
```

## Deployment Methods

### Method 1: Using Deployment Script (Recommended)

```bash
cd Part7

# Make scripts executable (Linux/Mac)
chmod +x deploy.sh undeploy.sh

# Deploy
./deploy.sh

# Undeploy
./undeploy.sh
```

### Method 2: Manual kubectl Commands

```bash
cd Part7

# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Apply ConfigMap
kubectl apply -f k8s/configmap.yaml -n mlops-heart-disease

# 3. Deploy application
kubectl apply -f k8s/deployment.yaml -n mlops-heart-disease

# 4. Apply HPA (optional)
kubectl apply -f k8s/hpa.yaml -n mlops-heart-disease

# 5. Apply Ingress (optional)
kubectl apply -f k8s/ingress.yaml -n mlops-heart-disease
```

### Method 3: Using Helm Chart

```bash
cd Part7/helm

# Install chart
helm install heart-disease-api . \
  --namespace mlops-heart-disease \
  --create-namespace

# Upgrade chart
helm upgrade heart-disease-api . \
  --namespace mlops-heart-disease

# Uninstall chart
helm uninstall heart-disease-api \
  --namespace mlops-heart-disease
```

## Kubernetes Resources

### 1. Namespace
Isolates the application resources.
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlops-heart-disease
```

### 2. Deployment
Manages 3 replicas of the API pods.
- **Image**: heart-disease-api:latest
- **Replicas**: 3
- **Resources**: 256Mi-512Mi memory, 250m-500m CPU
- **Health Checks**: Liveness and readiness probes

### 3. Service
Exposes the deployment via LoadBalancer.
- **Type**: LoadBalancer
- **Port**: 80 → 8000
- **Session Affinity**: ClientIP

### 4. ConfigMap
Stores configuration data.
- API settings
- Model path
- Logging configuration

### 5. HorizontalPodAutoscaler
Automatically scales pods based on metrics.
- **Min Replicas**: 2
- **Max Replicas**: 10
- **Target CPU**: 70%
- **Target Memory**: 80%

### 6. Ingress
Routes external traffic to the service.
- **Host**: heart-disease-api.local
- **Path**: /
- **Backend**: heart-disease-api-service:80

## Accessing the Deployed API

### For Minikube

```bash
# Get service URL
minikube service heart-disease-api-service -n mlops-heart-disease --url

# Or use port forwarding
kubectl port-forward service/heart-disease-api-service 8000:80 -n mlops-heart-disease

# Access API
curl http://localhost:8000/health
```

### For Cloud Deployment

```bash
# Get external IP
kubectl get service heart-disease-api-service -n mlops-heart-disease

# Wait for EXTERNAL-IP to be assigned
# Then access via:
curl http://<EXTERNAL-IP>/health
```

### Using Ingress

```bash
# For Minikube, add to /etc/hosts (or C:\Windows\System32\drivers\etc\hosts on Windows)
echo "$(minikube ip) heart-disease-api.local" | sudo tee -a /etc/hosts

# Access via hostname
curl http://heart-disease-api.local/health
```

## Monitoring Deployment

### Check Deployment Status
```bash
kubectl get deployments -n mlops-heart-disease
kubectl get pods -n mlops-heart-disease
kubectl get services -n mlops-heart-disease
kubectl get hpa -n mlops-heart-disease
```

### View Logs
```bash
# All pods
kubectl logs -f deployment/heart-disease-api -n mlops-heart-disease

# Specific pod
kubectl logs -f <pod-name> -n mlops-heart-disease

# Previous pod logs (if crashed)
kubectl logs <pod-name> -n mlops-heart-disease --previous
```

### Describe Resources
```bash
kubectl describe deployment heart-disease-api -n mlops-heart-disease
kubectl describe pod <pod-name> -n mlops-heart-disease
kubectl describe service heart-disease-api-service -n mlops-heart-disease
```

### Execute Commands in Pod
```bash
kubectl exec -it <pod-name> -n mlops-heart-disease -- /bin/bash
```

## Testing the Deployed API

### Health Check
```bash
curl http://<SERVICE-URL>/health
```

### Model Info
```bash
curl http://<SERVICE-URL>/model/info
```

### Prediction
```bash
curl -X POST http://<SERVICE-URL>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 -p sample_input.json -T application/json \
  http://<SERVICE-URL>/predict

# Using hey
hey -n 1000 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d @sample_input.json \
  http://<SERVICE-URL>/predict
```

## Scaling

### Manual Scaling
```bash
# Scale to 5 replicas
kubectl scale deployment heart-disease-api --replicas=5 -n mlops-heart-disease

# Verify
kubectl get pods -n mlops-heart-disease
```

### Auto-scaling with HPA
HPA automatically scales based on CPU/memory usage.
```bash
# Check HPA status
kubectl get hpa -n mlops-heart-disease

# Describe HPA
kubectl describe hpa heart-disease-api-hpa -n mlops-heart-disease
```

## Updating the Deployment

### Update Image
```bash
# Build new image
docker build -t heart-disease-api:v2 -f Part6/Dockerfile .

# For Minikube, load image
minikube image load heart-disease-api:v2

# Update deployment
kubectl set image deployment/heart-disease-api \
  api=heart-disease-api:v2 \
  -n mlops-heart-disease

# Check rollout status
kubectl rollout status deployment/heart-disease-api -n mlops-heart-disease
```

### Rollback Deployment
```bash
# Rollback to previous version
kubectl rollout undo deployment/heart-disease-api -n mlops-heart-disease

# Rollback to specific revision
kubectl rollout undo deployment/heart-disease-api --to-revision=2 -n mlops-heart-disease

# Check rollout history
kubectl rollout history deployment/heart-disease-api -n mlops-heart-disease
```

## Troubleshooting

### Pods Not Starting
```bash
# Check pod status
kubectl get pods -n mlops-heart-disease

# Describe pod
kubectl describe pod <pod-name> -n mlops-heart-disease

# Check events
kubectl get events -n mlops-heart-disease --sort-by='.lastTimestamp'
```

### Image Pull Errors
```bash
# For Minikube, ensure image is loaded
minikube image ls | grep heart-disease-api

# Load image if missing
minikube image load heart-disease-api:latest
```

### Service Not Accessible
```bash
# Check service
kubectl get service heart-disease-api-service -n mlops-heart-disease

# Check endpoints
kubectl get endpoints heart-disease-api-service -n mlops-heart-disease

# Port forward for testing
kubectl port-forward service/heart-disease-api-service 8000:80 -n mlops-heart-disease
```

### HPA Not Working
```bash
# Check if metrics-server is running
kubectl get deployment metrics-server -n kube-system

# For Minikube, enable metrics-server
minikube addons enable metrics-server

# Check metrics
kubectl top pods -n mlops-heart-disease
kubectl top nodes
```

## Production Best Practices

### 1. Resource Limits
Always set resource requests and limits to prevent resource exhaustion.

### 2. Health Checks
Configure liveness and readiness probes for automatic recovery.

### 3. Multiple Replicas
Run at least 3 replicas for high availability.

### 4. Auto-scaling
Use HPA to handle variable load automatically.

### 5. Monitoring
Integrate with Prometheus and Grafana for observability.

### 6. Logging
Use centralized logging (ELK stack, Loki, etc.).

### 7. Security
- Use network policies
- Enable RBAC
- Scan images for vulnerabilities
- Use secrets for sensitive data

### 8. CI/CD Integration
Automate deployment with GitHub Actions, GitLab CI, or Jenkins.

## Cleanup

### Remove All Resources
```bash
cd Part7
./undeploy.sh
```

### Or manually:
```bash
kubectl delete namespace mlops-heart-disease
```

### Stop Minikube
```bash
minikube stop
minikube delete
```

## Architecture Diagram

```
                                    ┌─────────────────┐
                                    │   LoadBalancer  │
                                    │   (Service)     │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
              ┌─────▼─────┐           ┌─────▼─────┐           ┌─────▼─────┐
              │   Pod 1   │           │   Pod 2   │           │   Pod 3   │
              │  FastAPI  │           │  FastAPI  │           │  FastAPI  │
              │   + Model │           │   + Model │           │   + Model │
              └───────────┘           └───────────┘           └───────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  ConfigMap      │
                                    │  (Configuration)│
                                    └─────────────────┘
```

## Next Steps
- Part 8: Monitoring & Logging with Prometheus and Grafana
- Part 9: Documentation & Reporting

