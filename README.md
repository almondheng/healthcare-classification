# Healthcare Classification

## Problem Description

Healthcare providers face the challenge of quickly and accurately identifying patients with critical conditions, such as cancer, from large volumes of patient data. Early and reliable detection is essential for timely treatment and improved patient outcomes.

This project addresses that challenge by building a machine learning model that predicts whether a patient has cancer based on a comprehensive set of features from the healthcare dataset. The dataset includes demographic information (age, gender, blood type), clinical details (admission type, medication, test results), and administrative data (insurance provider, hospital, doctor, billing amount, room number, etc.).

By automating the classification of cancer cases, this solution helps healthcare professionals prioritize care, optimize resource allocation, and support data-driven decision-making in clinical settings.


## Prerequisites

Python 3.10+

## Setup Instructions

### 1. Cloud Infrastructure (AWS S3 for MLflow)
- Install Terraform: https://www.terraform.io/downloads.html
- Configure AWS credentials.
- Provision S3 bucket for MLflow artifacts:
  ```cmd
  cd iac
  terraform init
  terraform apply
  ```

### 2. Install Python Dependencies
```cmd
python -m venv .venv/

# Activate virtual environment
source .venv/bin/activate # Unix/Linux/Mac
.venv\Scripts\activate # Windows

pip install -r requirements.txt
```

### 3. Download Dataset
- Download the dataset from Kaggle: https://www.kaggle.com/datasets/prasad22/healthcare-dataset
- Place `healthcare_dataset.csv` in a `data` folder at the project root.

### 4. Experiment Tracking & Model Registry
- Start MLflow tracking server (local or remote):
  ```cmd
mlflow ui
  ```
- Training logs metrics and model to MLflow.

### 5. Run Workflow Orchestration
```cmd
python src/workflow.py
```

### 6. Model Deployment
- Build Docker image:
  ```cmd
  docker build -t healthcare-classification .
  ```
- Run FastAPI server:
  ```cmd
  docker run -p 8000:8000 healthcare-classification
  ```

### 7. Model Monitoring
```cmd
python src/monitor.py
```

### 8. Testing & Code Quality
- Run unit/integration tests:
  ```cmd
  pytest tests
  ```
- Lint and format code:
  ```cmd
  flake8 src tests
  black src tests
  ```
- Install pre-commit hooks:
  ```cmd
  pre-commit install
  ```

### 9. CI/CD
- GitHub Actions workflow is set up in `.github/workflows/ci.yml`.
