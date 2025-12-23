#!/usr/bin/env python3
"""
FastAPI application for Heart Disease Prediction.
Serves the trained ML model via REST API with /predict endpoint.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
# Handle both local development and Docker container paths
if Path("/app/models/final_model.joblib").exists():
    # Running in Docker container
    PROJECT_ROOT = Path("/app")
else:
    # Running locally
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
sys.path.insert(0, str(PROJECT_ROOT))

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML model API for predicting heart disease risk based on patient health data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model variable
model = None
model_metadata = {}


class PatientData(BaseModel):
    """Input schema for patient health data."""
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: int = Field(..., ge=3, le=7, description="Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class BatchPatientData(BaseModel):
    """Input schema for batch predictions."""
    patients: List[PatientData] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Output schema for single prediction."""
    prediction: int = Field(..., description="Predicted class (0=no disease, 1=disease)")
    probability: float = Field(..., description="Probability of having heart disease")
    confidence: float = Field(..., description="Confidence score (max probability)")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionResponse]
    count: int


def load_model():
    """Load the trained model from Part4."""
    global model, model_metadata
    
    # Try Docker path first, then local path
    docker_model_path = Path("/app/models/final_model.joblib")
    local_model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
    
    if docker_model_path.exists():
        model_path = docker_model_path
        metadata_path = Path("/app/metrics/final_report.json")
        logger.info("Using Docker container paths")
    elif local_model_path.exists():
        model_path = local_model_path
        metadata_path = PROJECT_ROOT / "Part4" / "metrics" / "final_report.json"
        logger.info("Using local development paths")
    else:
        logger.error(f"Model not found at {docker_model_path} or {local_model_path}")
        raise FileNotFoundError(f"Model file not found in Docker or local paths")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Load metadata
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Model metadata loaded successfully")
        else:
            logger.warning(f"Metadata file not found at {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Heart Disease Prediction API...")
    load_model()
    logger.info("API ready to serve predictions")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Get model information and metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": model_metadata.get("chosen_model", "unknown"),
        "model_loaded": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add metrics if available
    if "metrics" in model_metadata:
        final_metrics = model_metadata["metrics"].get("final", {})
        if "cv" in final_metrics:
            info["performance"] = {
                "accuracy": final_metrics["cv"].get("accuracy_mean"),
                "precision": final_metrics["cv"].get("precision_mean"),
                "recall": final_metrics["cv"].get("recall_mean"),
                "roc_auc": final_metrics["cv"].get("roc_auc_mean")
            }
    
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData, request: Request):
    """
    Predict heart disease risk for a single patient.
    
    Returns prediction (0/1), probability, confidence, and risk level.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([patient.dict()])
        
        # Log request
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Prediction request from {client_host} - Age: {patient.age}, Sex: {patient.sex}")
        
        # Make prediction
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        
        # Get probability of positive class (disease)
        prob_disease = float(probabilities[1])
        confidence = float(max(probabilities))
        risk_level = get_risk_level(prob_disease)
        
        # Log prediction
        logger.info(f"Prediction: {prediction}, Probability: {prob_disease:.3f}, Risk: {risk_level}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=prob_disease,
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(batch: BatchPatientData, request: Request):
    """
    Predict heart disease risk for multiple patients.
    
    Accepts up to 100 patients in a single request.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([p.dict() for p in batch.patients])
        
        # Log request
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Batch prediction request from {client_host} - Count: {len(batch.patients)}")
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Build response
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            prob_disease = float(probs[1])
            confidence = float(max(probs))
            risk_level = get_risk_level(prob_disease)
            
            results.append(PredictionResponse(
                prediction=int(pred),
                probability=prob_disease,
                confidence=confidence,
                risk_level=risk_level,
                timestamp=datetime.utcnow().isoformat()
            ))
        
        logger.info(f"Batch prediction completed - {len(results)} predictions")
        
        return BatchPredictionResponse(
            predictions=results,
            count=len(results)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

