# Multi-stage Dockerfile for Heart Disease Prediction API
# Stage 1: Base image with dependencies
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy installed packages from base
COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY Part6/src/app.py /app/
COPY Part4/models/final_model.joblib /app/models/
COPY Part4/metrics/final_report.json /app/metrics/

# Copy necessary modules for imports
COPY Part2/src/features.py /app/Part2/src/

# Create necessary directories
RUN mkdir -p /app/Part2/src /app/Part4/models /app/Part4/metrics

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/final_model.joblib

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

