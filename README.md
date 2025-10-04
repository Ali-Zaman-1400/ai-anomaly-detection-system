# AI Anomaly Detection System
A production-style project that detects anomalies in numeric data using Isolation Forest and Autoencoders. Includes data generation, preprocessing, training, evaluation, and a FastAPI service.

## Quickstart
1) `pip install -r requirements.txt`
2) `python src/generate_data.py`
3) `python src/preprocess.py`
4) `python src/train_iforest.py` (optional: `python src/train_autoencoder.py`)
5) `python src/evaluate.py`
6) `uvicorn src.api:app --reload`

Adjust `config/settings.yaml` to change parameters or switch models.
