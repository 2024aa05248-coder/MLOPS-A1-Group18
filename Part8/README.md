# Part 8 — Monitoring & Logging

## Scope
- Comprehensive logging of all API requests and predictions
- Prometheus metrics collection
- Grafana dashboards for visualization
- Real-time monitoring of API performance
- Error tracking and alerting

## Features

### 1. Comprehensive Logging
- **Request Logging**: All HTTP requests with timestamps, client info, and response times
- **Prediction Logging**: Every prediction with input features, results, and confidence scores
- **Error Logging**: Detailed error messages with stack traces
- **Structured Logging**: JSON-formatted logs for easy parsing

### 2. Prometheus Metrics
- **Request Metrics**:
  - `api_requests_total`: Total number of requests by endpoint and status
  - `api_request_duration_seconds`: Request duration histogram
  - `active_requests`: Current number of active requests

- **Prediction Metrics**:
  - `predictions_total`: Total predictions by class and risk level
  - `prediction_duration_seconds`: Prediction processing time

- **Model Metrics**:
  - `model_load_time_seconds`: Time taken to load the model

- **Error Metrics**:
  - `api_errors_total`: Total errors by type and endpoint

### 3. Grafana Dashboards
- Real-time API performance visualization
- Request rate and latency graphs
- Prediction distribution by risk level
- Error rate monitoring
- Active requests tracking

## Setup and Running

### Prerequisites
```bash
# Install monitoring dependencies
pip install prometheus-client

# Or install from requirements.txt
pip install -r requirements.txt
```

### Option 1: Run API with Monitoring Locally

```bash
cd Part8/src
python app_with_monitoring.py
```

Access:
- **API**: http://localhost:8000
- **Metrics**: http://localhost:8000/metrics
- **Docs**: http://localhost:8000/docs

### Option 2: Run Full Monitoring Stack with Docker Compose

```bash
cd Part8
docker-compose -f docker-compose-monitoring.yml up -d
```

Access:
- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Stop Monitoring Stack
```bash
cd Part8
docker-compose -f docker-compose-monitoring.yml down
```

## Viewing Logs

### Application Logs
```bash
# Real-time logs
tail -f api.log

# Filter by level
grep "ERROR" api.log
grep "WARNING" api.log

# View in Docker
docker logs -f heart-disease-api-monitored
```

### Log Format
```json
{
  "timestamp": "2025-12-22T18:30:00.000Z",
  "method": "POST",
  "url": "/predict",
  "client_host": "172.17.0.1",
  "user_agent": "curl/7.68.0",
  "response": {
    "prediction": 1,
    "risk_level": "High"
  }
}
```

## Prometheus Metrics

### Access Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```

### Example Metrics Output
```
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="200"} 150.0

# HELP api_request_duration_seconds API request duration in seconds
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="0.1"} 120.0
api_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="0.5"} 145.0
api_request_duration_seconds_sum{endpoint="/predict",method="POST"} 45.2
api_request_duration_seconds_count{endpoint="/predict",method="POST"} 150.0

# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total{prediction_class="0",risk_level="Low"} 80.0
predictions_total{prediction_class="1",risk_level="High"} 70.0
```

### Prometheus Queries

#### Request Rate
```promql
rate(api_requests_total[5m])
```

#### Average Request Duration
```promql
rate(api_request_duration_seconds_sum[5m]) / rate(api_request_duration_seconds_count[5m])
```

#### 95th Percentile Latency
```promql
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))
```

#### Error Rate
```promql
rate(api_errors_total[5m])
```

#### Predictions by Risk Level
```promql
sum by (risk_level) (predictions_total)
```

## Grafana Dashboards

### Setup Grafana

1. **Access Grafana**: http://localhost:3000
2. **Login**: admin/admin (change password on first login)
3. **Add Prometheus Data Source**:
   - Go to Configuration → Data Sources
   - Add Prometheus
   - URL: http://prometheus:9090
   - Save & Test

4. **Import Dashboard**:
   - Go to Create → Import
   - Upload `dashboards/api-dashboard.json`
   - Select Prometheus data source
   - Import

### Dashboard Panels

1. **Total Requests**: Total number of API requests
2. **Request Rate**: Requests per second by endpoint
3. **Request Duration**: 95th percentile latency
4. **Total Predictions**: Total predictions made
5. **Predictions by Risk Level**: Pie chart of risk distribution
6. **Active Requests**: Current active requests
7. **Error Rate**: Errors per second by type

### Creating Custom Panels

1. Click "Add Panel"
2. Select visualization type (Graph, Stat, Gauge, etc.)
3. Enter Prometheus query
4. Configure display options
5. Save dashboard

## Monitoring Best Practices

### 1. Set Up Alerts

Create alerts in Prometheus (`alerting_rules.yml`):

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(api_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency"
          description: "95th percentile latency is {{ $value }}s"
```

### 2. Log Rotation

Configure log rotation to prevent disk space issues:

```bash
# /etc/logrotate.d/api-logs
/app/logs/api.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
}
```

### 3. Centralized Logging

For production, use centralized logging:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Loki + Grafana**
- **Cloud logging** (CloudWatch, Stackdriver, Azure Monitor)

### 4. Monitoring Checklist

- Request rate and latency
- Error rate and types
- Prediction distribution
- Resource usage (CPU, memory)
- Model performance metrics
- Health check status
- Active connections

## Troubleshooting

### Metrics Not Showing
```bash
# Check if metrics endpoint is accessible
curl http://localhost:8000/metrics

# Check Prometheus targets
# Go to http://localhost:9090/targets
```

### Grafana Can't Connect to Prometheus
```bash
# Check if Prometheus is running
docker ps | grep prometheus

# Check network connectivity
docker exec grafana ping prometheus

# Verify Prometheus URL in Grafana data source
```

### High Memory Usage
```bash
# Check container stats
docker stats

# Limit Prometheus retention
# Add to prometheus.yml:
# --storage.tsdb.retention.time=7d
```

## Integration with Kubernetes

### ServiceMonitor for Prometheus Operator

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: heart-disease-api
  labels:
    app: heart-disease-api
spec:
  selector:
    matchLabels:
      app: heart-disease-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### Deploy Prometheus in Kubernetes

```bash
# Using Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

## Sample Monitoring Queries

### API Health
```bash
# Check if API is up
up{job="heart-disease-api"}

# Request success rate
sum(rate(api_requests_total{status="200"}[5m])) / sum(rate(api_requests_total[5m]))
```

### Performance
```bash
# Average prediction time
rate(prediction_duration_seconds_sum[5m]) / rate(prediction_duration_seconds_count[5m])

# Requests per minute
sum(rate(api_requests_total[1m])) * 60
```

### Business Metrics
```bash
# High-risk predictions percentage
sum(predictions_total{risk_level="High"}) / sum(predictions_total) * 100

# Predictions per hour
increase(predictions_total[1h])
```

## Next Steps
- Part 9: Final documentation and reporting
- Set up alerting with AlertManager
- Integrate with incident management tools (PagerDuty, Opsgenie)
- Create custom dashboards for business metrics

