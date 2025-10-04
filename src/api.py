from fastapi import FastAPI
from pydantic import BaseModel, conlist
import numpy as np, joblib
from tensorflow import keras
from utils import load_config
app = FastAPI(title='Anomaly Detection API')
cfg = load_config()
model_choice = cfg['api']['model']
SCALER = joblib.load('data/models/scaler.joblib')
if model_choice == 'iforest':
    MODEL = joblib.load(cfg['models']['iforest']['model_path'])
elif model_choice == 'autoencoder':
    MODEL = keras.models.load_model(cfg['models']['autoencoder']['model_path'])
else:
    raise ValueError('Unknown model selected in config.api.model')
class PredictRequest(BaseModel):
    features: conlist(float, min_length=1)
class PredictResponse(BaseModel):
    is_anomaly: bool
    score: float
@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.array(req.features, dtype=float).reshape(1, -1)
    x = SCALER.transform(x)
    if model_choice == 'iforest':
        score = -float(MODEL.score_samples(x)[0])
        is_anom = bool(MODEL.predict(x)[0] == -1)
        return PredictResponse(is_anomaly=is_anom, score=score)
    else:
        recon = MODEL.predict(x, verbose=0)
        mse = float(np.mean((x - recon)**2))
        is_anom = mse > 0.5
        return PredictResponse(is_anomaly=is_anom, score=mse)
