# Part 6 — Model Containerization

## Scope
- FastAPI application serving the trained ML model
- REST API with `/predict` endpoint for single and batch predictions
- Docker containerization for isolated, reproducible deployment
- Health checks and monitoring endpoints

## API Endpoints

### 1. Root (`GET /`)
Returns API information and available endpoints.

### 2. Health Check (`GET /health`)
Health check endpoint for monitoring and orchestration.

### 3. Model Info (`GET /model/info`)
Returns model metadata and performance metrics.

### 4. Predict (`POST /predict`)
Single patient prediction endpoint.

**Request Body:**
```json
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
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.234,
  "confidence": 0.766,
  "risk_level": "Low",
  "timestamp": "2025-12-22T18:30:00.000Z"
}
```

### 5. Batch Predict (`POST /batch_predict`)
Batch prediction endpoint (up to 100 patients).

**Request Body:**
```json
{
  "patients": [
    { /* patient 1 data */ },
    { /* patient 2 data */ }
  ]
}
```

## Running Locally

### Prerequisites
- Python 3.9+
- Model trained and packaged (Part 4 completed)

### Install Dependencies
```bash
pip install fastapi uvicorn pydantic python-multipart
```

### Run API Server
```bash
cd Part6/src
python app.py
```

Or with uvicorn directly:
```bash
cd Part6/src
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Access API
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Test API
```bash
cd Part6/src
python test_api.py
```

## Docker Containerization

### Build Docker Image
```bash
# From project root
docker build -t heart-disease-api:latest -f Part6/Dockerfile .
```

### Run Docker Container
```bash
docker run -d \
  --name heart-disease-api \
  -p 8000:8000 \
  heart-disease-api:latest
```

### Check Container Status
```bash
docker ps
docker logs heart-disease-api
```

### Test Containerized API
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

### Stop and Remove Container
```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

## Docker Compose

### Run with Docker Compose
```bash
cd Part6
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f
```

### Stop Services
```bash
docker-compose down
```

## API Features

### Input Validation
- Pydantic models ensure type safety and validation
- Range checks for all numeric inputs
- Clear error messages for invalid inputs

### Error Handling
- Global exception handler
- Detailed error logging
- HTTP status codes for different error types

### Logging
- Request logging with timestamps
- Prediction logging for monitoring
- Error and exception logging

### Health Checks
- Docker health check configured
- Kubernetes-ready health endpoint
- Model loading status verification

## Testing

### Manual Testing with curl
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

### Automated Testing
```bash
cd Part6/src
python test_api.py
```

### Load Testing (Optional)
```bash
# Using Apache Bench
ab -n 1000 -c 10 -p sample_input.json -T application/json \
  http://localhost:8000/predict
```

## Production Considerations

### Security
- Input validation prevents injection attacks
- No secrets in code (use environment variables)
- HTTPS should be configured at load balancer level

### Performance
- Async FastAPI for concurrent requests
- Model loaded once at startup
- Efficient numpy/pandas operations

### Monitoring
- Health check endpoint for orchestration
- Structured logging for analysis
- Request/response logging for debugging

### Scalability
- Stateless design allows horizontal scaling
- Container-ready for orchestration
- Can be deployed behind load balancer

## Troubleshooting

### Model Not Found
Ensure Part 4 is completed and model exists at:
```
Part4/models/final_model.joblib
```

### Port Already in Use
Change port in docker run command:
```bash
docker run -p 8080:8000 heart-disease-api:latest
```

### Container Won't Start
Check logs:
```bash
docker logs heart-disease-api
```

### Import Errors
Ensure all dependencies are in requirements.txt and installed in container.

