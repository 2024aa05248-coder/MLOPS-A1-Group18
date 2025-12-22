#!/usr/bin/env python3
"""
Test script for the FastAPI application.
Can be run locally to test API endpoints.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint."""
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_health():
    """Test health check endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_model_info():
    """Test model info endpoint."""
    print("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_predict():
    """Test single prediction endpoint."""
    print("Testing predict endpoint...")
    
    # Sample patient data
    patient_data = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(patient_data, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_predict():
    """Test batch prediction endpoint."""
    print("Testing batch predict endpoint...")
    
    # Sample batch data
    batch_data = {
        "patients": [
            {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6
            },
            {
                "age": 67,
                "sex": 1,
                "cp": 4,
                "trestbps": 160,
                "chol": 286,
                "fbs": 0,
                "restecg": 2,
                "thalach": 108,
                "exang": 1,
                "oldpeak": 1.5,
                "slope": 2,
                "ca": 3,
                "thal": 3
            },
            {
                "age": 45,
                "sex": 0,
                "cp": 3,
                "trestbps": 130,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 170,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 0,
                "thal": 3
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=batch_data)
    print(f"Status: {response.status_code}")
    print(f"Request: {len(batch_data['patients'])} patients")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Heart Disease Prediction API - Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_predict()
        test_batch_predict()
        
        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API. Make sure it's running on", BASE_URL)
    except Exception as e:
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()

