# Heart Disease Prediction (Task 1: Data Acquisition & EDA)

## Objective
Design, develop, and deploy a scalable and reproducible ML solution using modern MLOps practices. This repository starts with Task 1: Data Acquisition and Exploratory Data Analysis (EDA) using the UCI Heart Disease dataset.

## Dataset
- **Source**: UCI Machine Learning Repository — Heart Disease Dataset
  - Homepage: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
  - Dataset used (for automation): processed.cleveland.data
- **Schema** (UCI Cleveland processed):
  - Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num
  - Target: num (0..4). We derive binary target: target = 1 if num > 0 else 0.

## What Task 1 Includes
- Automated dataset download (scripted, reproducible)
- Cleaning and preprocessing:
  - Replace '?' with NaN
  - Type casting
  - Numeric imputation (median)
  - Categorical imputation (most frequent)
  - Feature encoding (one-hot for categorical features)
  - Binary target derivation
- Professional EDA visualizations:
  - Histograms for numerical features
  - Class balance bar chart
  - Correlation heatmap (including target)

## Project Structure
```
MLOP-Assign/
└── Part1/
    ├── README.md
    ├── dataset_download_script/
    │   └── download_data.py
    ├── src/
    │   ├── data_preprocess.py
    │   └── eda.py
    ├── data/
    │   ├── raw/               # downloaded raw dataset from UCI
    │   ├── interim/           # cleaned, imputed but not encoded
    │   └── processed/         # fully encoded features + target
    └── reports/
        └── figures/           # saved EDA plots
```

## Prerequisites
- Python 3.9+ (tested on macOS & Windows)
- pip

## Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## How to Run

### 1) Download Dataset (from UCI)
```bash
python dataset_download_script/download_data.py
```

This downloads:
- `data/raw/processed.cleveland.data`

### 2) Clean and Preprocess
```bash
python src/data_preprocess.py
```

This produces:
- `data/interim/heart_clean.csv` (imputed, typed, with target)
- `data/processed/heart_encoded.csv` (one-hot encoded feature matrix + target)
- `data/processed/feature_names.json` (encoded feature names for reference)

### 3) Run EDA
```bash
python src/eda.py
```

This saves plots to:
- `reports/figures/histograms_numeric.png`
- `reports/figures/class_balance.png`
- `reports/figures/corr_heatmap.png`

## Preprocessing Details

### Missing Values:
- `?` to NaN
- Numeric columns: median imputation
- Categorical columns: most frequent imputation

### Encoding:
- OneHotEncoder for categorical: cp, restecg, slope, thal

### Target:
- Derived as binary from UCI num: `target = (num > 0).astype(int)`

## Next Steps

Proceed to Part 2: Model Training and Evaluation