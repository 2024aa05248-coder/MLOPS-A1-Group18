#!/usr/bin/env python3
"""
FastAPI application with comprehensive logging and Prometheus metrics.
Enhanced version of Part6 API with monitoring capabilities.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
from functools import wraps

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
import uvicorn

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API with Monitoring",
    description="ML model API with comprehensive logging and Prometheus metrics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['prediction_class', 'risk_level']
)

PREDICTION_DURATION = Histogram(
    'prediction_duration_seconds',
    'Prediction processing time in seconds'
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time taken to load the model'
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests being processed'
)

ERROR_COUNT = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['error_type', 'endpoint']
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
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels")
    thal: int = Field(..., ge=3, le=7, description="Thalassemia")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
            }
        }


class BatchPatientData(BaseModel):
    """Input schema for batch predictions."""
    patients: List[PatientData] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Output schema for single prediction."""
    prediction: int
    probability: float
    confidence: float
    risk_level: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionResponse]
    count: int


def load_model():
    """Load the trained model from Part4."""
    global model, model_metadata
    
    start_time = time.time()
    model_path = PROJECT_ROOT / "Part4" / "models" / "final_model.joblib"
    
    logger.info(f"Loading model from {model_path}")
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = joblib.load(model_path)
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Load metadata
        metadata_path = PROJECT_ROOT / "Part4" / "metrics" / "final_report.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Model metadata loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        ERROR_COUNT.labels(error_type='model_load_error', endpoint='startup').inc()
        raise


def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def log_request(request: Request, response_data: dict = None):
    """Log request details."""
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "method": request.method,
        "url": str(request.url),
        "client_host": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
    }
    
    if response_data:
        log_data["response"] = response_data
    
    logger.info(f"Request: {json.dumps(log_data)}")


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware to monitor all requests."""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration:.3f}s"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        ERROR_COUNT.labels(error_type='request_error', endpoint=request.url.path).inc()
        raise
    
    finally:
        ACTIVE_REQUESTS.dec()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("=" * 60)
    logger.info("Starting Heart Disease Prediction API with Monitoring")
    logger.info("=" * 60)
    load_model()
    logger.info("API ready to serve predictions")
    logger.info("=" * 60)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Heart Disease Prediction API with Monitoring",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.debug(f"Health check: {health_status}")
    return health_status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    logger.debug("Metrics endpoint called")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
async def model_info():
    """Get model information and metadata."""
    if model is None:
        logger.warning("Model info requested but model not loaded")
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
    
    logger.info(f"Model info retrieved: {info['model_type']}")
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData, request: Request):
    """Predict heart disease risk for a single patient."""
    if model is None:
        logger.error("Prediction attempted but model not loaded")
        ERROR_COUNT.labels(error_type='model_not_loaded', endpoint='/predict').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([patient.dict()])
        
        # Log request
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            f"Prediction request from {client_host} - "
            f"Patient: Age={patient.age}, Sex={patient.sex}, CP={patient.cp}"
        )
        
        # Make prediction
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        
        # Get probability of positive class (disease)
        prob_disease = float(probabilities[1])
        confidence = float(max(probabilities))
        risk_level = get_risk_level(prob_disease)
        
        # Record metrics
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)
        PREDICTION_COUNT.labels(
            prediction_class=str(prediction),
            risk_level=risk_level
        ).inc()
        
        # Log prediction
        logger.info(
            f"Prediction completed - "
            f"Result: {prediction}, "
            f"Probability: {prob_disease:.3f}, "
            f"Risk: {risk_level}, "
            f"Duration: {duration:.3f}s"
        )
        
        response = PredictionResponse(
            prediction=prediction,
            probability=prob_disease,
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.utcnow().isoformat()
        )
        
        log_request(request, {"prediction": prediction, "risk_level": risk_level})
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        ERROR_COUNT.labels(error_type='prediction_error', endpoint='/predict').inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(batch: BatchPatientData, request: Request):
    """Predict heart disease risk for multiple patients."""
    if model is None:
        logger.error("Batch prediction attempted but model not loaded")
        ERROR_COUNT.labels(error_type='model_not_loaded', endpoint='/batch_predict').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
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
            
            # Record metrics for each prediction
            PREDICTION_COUNT.labels(
                prediction_class=str(pred),
                risk_level=risk_level
            ).inc()
            
            results.append(PredictionResponse(
                prediction=int(pred),
                probability=prob_disease,
                confidence=confidence,
                risk_level=risk_level,
                timestamp=datetime.utcnow().isoformat()
            ))
        
        # Record batch metrics
        duration = time.time() - start_time
        PREDICTION_DURATION.observe(duration)
        
        logger.info(
            f"Batch prediction completed - "
            f"Count: {len(results)}, "
            f"Duration: {duration:.3f}s, "
            f"Avg per prediction: {duration/len(results):.3f}s"
        )
        
        return BatchPredictionResponse(
            predictions=results,
            count=len(results)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        ERROR_COUNT.labels(error_type='batch_prediction_error', endpoint='/batch_predict').inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True
    )
    ERROR_COUNT.labels(error_type='unhandled_exception', endpoint=request.url.path).inc()
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
        "app_with_monitoring:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

