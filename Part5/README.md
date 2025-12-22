# Part 5 — CI/CD Pipeline & Automated Testing

## Scope
- Comprehensive unit test suite for data processing, feature engineering, model training, and inference
- GitHub Actions CI/CD pipeline with linting, testing, and model training
- Automated quality checks and artifact generation

## Test Suite

### Test Files
- `tests/test_data_preprocessing.py` - Tests for data cleaning, imputation, and validation
- `tests/test_features.py` - Tests for feature engineering and transformation pipelines
- `tests/test_model.py` - Tests for model training, evaluation, and persistence
- `tests/test_inference.py` - Tests for inference pipeline and predictions

### Running Tests

#### Run all tests
```bash
cd Part5
pytest
```

#### Run specific test file
```bash
pytest tests/test_data_preprocessing.py -v
```

#### Run tests with coverage
```bash
pytest --cov=../Part1 --cov=../Part2 --cov=../Part4 --cov-report=html
```

#### Run tests matching a pattern
```bash
pytest -k "test_model" -v
```

## CI/CD Pipeline

### GitHub Actions Workflow
Located at `.github/workflows/ci-cd.yml`

**Pipeline Stages:**
1. **Linting** - Code quality checks with flake8
2. **Testing** - Run full test suite with pytest
3. **Model Training** - Train models and log artifacts
4. **Artifact Upload** - Save models, metrics, and plots

### Workflow Triggers
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual workflow dispatch

### Environment
- Python 3.9
- All dependencies from `requirements.txt`
- Cached pip packages for faster builds

## Test Coverage

The test suite covers:
- ✅ Data preprocessing and cleaning
- ✅ Missing value handling
- ✅ Feature engineering pipelines
- ✅ Model training (Logistic Regression, Random Forest)
- ✅ Cross-validation
- ✅ Metrics calculation (accuracy, precision, recall, ROC-AUC)
- ✅ Model persistence (save/load)
- ✅ Inference pipeline
- ✅ Input validation
- ✅ Output format verification

## Prerequisites

Install testing dependencies:
```bash
pip install pytest pytest-cov flake8 black
```

## Continuous Integration

### Local CI Simulation
Run the same checks that CI runs:

```bash
# Linting
flake8 Part1 Part2 Part3 Part4 --max-line-length=120 --extend-ignore=E203,W503

# Testing
cd Part5
pytest -v

# Code formatting check (optional)
black --check Part1 Part2 Part3 Part4 Part5
```

## Artifacts

CI pipeline generates:
- Test results (JUnit XML)
- Coverage reports (HTML)
- Trained models (joblib files)
- Metrics (JSON files)
- Plots (PNG files)

All artifacts are uploaded and accessible from GitHub Actions run page.

## Notes
- Tests use fixtures for sample data to ensure reproducibility
- Tests are independent and can run in any order
- Mock data is used where real data isn't available
- Integration tests check actual model files from Part2 and Part4

