#!/bin/bash
# Cleanup script for Heart Disease Prediction API on Kubernetes

set -e

echo "========================================"
echo "Heart Disease API - Kubernetes Cleanup"
echo "========================================"
echo ""

NAMESPACE="mlops-heart-disease"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl is not installed."
    exit 1
fi

echo "Deleting all resources in namespace ${NAMESPACE}..."
echo ""

# Delete HPA
kubectl delete -f Part7/k8s/hpa.yaml -n ${NAMESPACE} --ignore-not-found=true
echo "✓ HPA deleted"

# Delete Ingress
kubectl delete -f Part7/k8s/ingress.yaml -n ${NAMESPACE} --ignore-not-found=true
echo "✓ Ingress deleted"

# Delete Deployment and Service
kubectl delete -f Part7/k8s/deployment.yaml -n ${NAMESPACE} --ignore-not-found=true
echo "✓ Deployment and Service deleted"

# Delete ConfigMap
kubectl delete -f Part7/k8s/configmap.yaml -n ${NAMESPACE} --ignore-not-found=true
echo "✓ ConfigMap deleted"

# Delete Namespace (optional - uncomment if you want to delete the namespace)
# kubectl delete -f Part7/k8s/namespace.yaml --ignore-not-found=true
# echo "✓ Namespace deleted"

echo ""
echo "========================================"
echo "Cleanup completed successfully!"
echo "========================================"

