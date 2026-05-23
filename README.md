# mlops-plc-cycle-prediction

This repository contains an MLOps-style PLC cycle-time prediction project. The original industrial cycle-time dataset is not included because it contains confidential production information. A synthetic PLC-style dataset is used instead to preserve the same machine-learning workflow without exposing private production data.

## Project summary

The pipeline performs:

- synthetic PLC event dataset generation
- preprocessing and cycle-level feature engineering
- abnormal cycle classification
- model training
- local MLflow experiment tracking
- Flask API model serving
- Docker image build
- GitHub Container Registry publication
- API validation through health and prediction endpoints

## Repository structure

```text
app/                  Flask prediction API
src/                  data generation, preprocessing, and training code
data/raw/             synthetic input dataset
tests/                API and pipeline tests
.github/workflows/    GitHub Actions workflow
Dockerfile            container definition
requirements.txt      Python dependencies
```
