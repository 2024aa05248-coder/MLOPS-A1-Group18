#!/bin/bash
# Deployment script for Heart Disease Prediction API on Kubernetes

set -e

echo "========================================"
echo "Heart Disease API - Kubernetes Deployment"
echo "========================================"
echo ""

# Configuration
NAMESPACE="mlops-heart-disease"
IMAGE_NAME="heart-disease-api"
IMAGE_TAG="latest"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "ERROR: Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

echo "[OK] kubectl is installed and cluster is accessible"
echo ""

# Build Docker image
echo "Step 1: Building Docker image..."
cd ..
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Part6/Dockerfile .
echo "[OK] Docker image built successfully"
echo ""

# For Minikube, load image into Minikube
if kubectl config current-context | grep -q "minikube"; then
    echo "Step 2: Loading image into Minikube..."
    minikube image load ${IMAGE_NAME}:${IMAGE_TAG}
    echo "[OK] Image loaded into Minikube"
    echo ""
fi

# Create namespace
echo "Step 3: Creating namespace..."
kubectl apply -f Part7/k8s/namespace.yaml
echo "[OK] Namespace created"
echo ""

# Apply ConfigMap
echo "Step 4: Applying ConfigMap..."
kubectl apply -f Part7/k8s/configmap.yaml -n ${NAMESPACE}
echo "[OK] ConfigMap applied"
echo ""

# Deploy application
echo "Step 5: Deploying application..."
kubectl apply -f Part7/k8s/deployment.yaml -n ${NAMESPACE}
echo "[OK] Deployment created"
echo ""

# Wait for deployment to be ready
echo "Step 6: Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/heart-disease-api -n ${NAMESPACE}
echo "[OK] Deployment is ready"
echo ""

# Apply HPA (optional)
echo "Step 7: Applying Horizontal Pod Autoscaler..."
kubectl apply -f Part7/k8s/hpa.yaml -n ${NAMESPACE} || echo "[WARNING] HPA not applied (metrics-server may not be installed)"
echo ""

# Get deployment status
echo "========================================"
echo "Deployment Status"
echo "========================================"
kubectl get deployments -n ${NAMESPACE}
echo ""
kubectl get pods -n ${NAMESPACE}
echo ""
kubectl get services -n ${NAMESPACE}
echo ""

# Get service URL
echo "========================================"
echo "Access Information"
echo "========================================"

if kubectl config current-context | grep -q "minikube"; then
    SERVICE_URL=$(minikube service heart-disease-api-service -n ${NAMESPACE} --url)
    echo "API URL: ${SERVICE_URL}"
    echo ""
    echo "To test the API:"
    echo "  curl ${SERVICE_URL}/health"
    echo "  curl ${SERVICE_URL}/model/info"
else
    echo "Service type: LoadBalancer"
    echo "Waiting for external IP..."
    kubectl get service heart-disease-api-service -n ${NAMESPACE}
    echo ""
    echo "To get the external IP:"
    echo "  kubectl get service heart-disease-api-service -n ${NAMESPACE}"
fi

echo ""
echo "========================================"
echo "Deployment completed successfully!"
echo "========================================"
echo ""
echo "Useful commands:"
echo "  kubectl get pods -n ${NAMESPACE}"
echo "  kubectl logs -f deployment/heart-disease-api -n ${NAMESPACE}"
echo "  kubectl describe deployment heart-disease-api -n ${NAMESPACE}"
echo "  kubectl port-forward service/heart-disease-api-service 8000:80 -n ${NAMESPACE}"

