# Heart Disease Prediction - MLOps Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-containerized-blue)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-ready-blue)](https://kubernetes.io/)

## Project Overview

A complete MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset. This project demonstrates end-to-end machine learning workflow from data acquisition to production deployment with comprehensive monitoring.

### Key Features

- **High Performance**: 91.84% ROC-AUC on heart disease prediction
- **Fully Automated**: CI/CD pipeline with GitHub Actions
- **Production Ready**: Docker containerization + Kubernetes orchestration
- **Experiment Tracking**: MLflow integration for reproducibility
- **Comprehensive Testing**: 40 unit tests with pytest
- **Real-time Monitoring**: Prometheus metrics + Grafana dashboards
- **Well Documented**: Extensive documentation and guides

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Model Performance** | 91.84% ROC-AUC |
| **API Latency** | <50ms (p95: 80ms) |
| **Test Coverage** | 40 unit tests |
| **Scalability** | 2-10 pods auto-scaling |
| **Availability** | 3 replicas, zero downtime |

---

## Quick Start


#### Step-by-Step Guide
```powershell
# Follow the complete guide
Get-Content COMPLETE_SETUP_GUIDE.md
```
**Time:** 60-100 minutes

---

### Prerequisites

- Python 3.9+
- Docker Desktop (with Kubernetes enabled)
- Minikube
- kubectl

### Installation

```bash
# Clone repository
git clone <repository-url>
cd MLOPS-A1-Group18

# Create virtual environment and activate
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Documentation

**[COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)** 
### Individual Part READMEs
- [Part 1 README](Part1/README.md) - Data & EDA
- [Part 2 README](Part2/README.md) - Model Development
- [Part 3 README](Part3/README.md) - Experiment Tracking
- [Part 4 README](Part4/README.md) - Model Packaging
- [Part 5 README](Part5/README.md) - CI/CD & Testing
- [Part 6 README](Part6/README.md) - Containerization
- [Part 7 README](Part7/README.md) - Deployment
- [Part 8 README](Part8/README.md) - Monitoring

---

## Project Structure

```
MLOP-Assign/
├── Part1/              # Data Acquisition & EDA
├── Part2/              # Feature Engineering & Models
├── Part3/              # Experiment Tracking
├── Part4/              # Model Packaging
├── Part5/              # CI/CD & Testing 
├── Part6/              # Containerization 
├── Part7/              # Production Deployment 
├── Part8/              # Monitoring & Logging 
├── Part9/              # Documentation & Reporting 
├── .github/workflows/  # CI/CD pipeline
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

Each part has its own README with detailed instructions.

---

## Model Performance

### Logistic Regression
- Accuracy: 85.49% ± 4.18%
- Precision: 85.94% ± 5.82%
- Recall: 82.04% ± 6.33%
- ROC-AUC: 91.80% ± 2.27%

### Random Forest (Selected)
- Accuracy: 82.84% ± 3.20%
- Precision: 83.42% ± 4.59%
- Recall: 78.39% ± 6.10%
- **ROC-AUC: 91.84% ± 1.79%**

---

## Technology Stack

### ML & Data
- scikit-learn, pandas, numpy, scipy
- MLflow (experiment tracking)

### API & Web
- FastAPI, Pydantic, Uvicorn

### Infrastructure
- Docker (containerization)
- Kubernetes (orchestration)
- Helm (package management)

### Monitoring
- Prometheus (metrics)
- Grafana (visualization)

### CI/CD
- GitHub Actions
- pytest, flake8, black

---

## Demo

For a live demonstration:

1. Follow the video link attached in the Final Report
2. Duration: 28 minutes
3. Covers all major components
4. Includes talking points and commands

---

## Troubleshooting

### Common Issues

**Model Not Found**
```bash
python Part4/src/package_model.py
```

**Port Already in Use**
```bash
# Use different port
docker run -p 8080:8000 heart-disease-api:latest
```

**Tests Failing**
```bash
# Ensure data is preprocessed
python Part1/src/data_preprocess.py
cd Part5 && pytest -v
```

---

## The Team

| Name                | Roll No.    | Contribution |
|---------------------|-------------|--------------|
| Ashmita De          | 2024AA05248 | 100%         |
| Ayush Goyal         | 2024AA05463 | 100%         |
| Srinivasan V        | 2024AA05292 | 100%         |
| Saurabh Vikas Kolhe | 2024AA05350 | 100%         |
---

## License

This project is for educational purposes as part of MLOps coursework.

---

## Acknowledgments

- **Dataset**: UCI Machine Learning Repository
- **Libraries**: scikit-learn, MLflow, FastAPI, and all open-source contributors
- **Tools**: Docker, Kubernetes, Prometheus, Grafana
- **Inspiration**: MLOps best practices from Google, Microsoft

---

**Built by Group18**

