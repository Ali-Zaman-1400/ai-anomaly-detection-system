import pandas as pd, numpy as np, joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from utils import load_config, setup_logging
from tensorflow import keras
def eval_iforest(cfg):
    X_test = pd.read_csv(f"{cfg['paths']['processed']}/X_test.csv").values
    y_test = pd.read_csv(f"{cfg['paths']['processed']}/y_test.csv")['is_anomaly'].values
    model = joblib.load(cfg['models']['iforest']['model_path'])
    preds = model.predict(X_test)
    y_pred = (preds == -1).astype(int)
    print('Isolation Forest results')
    print(classification_report(y_test, y_pred, digits=4))
    try:
        scores = -model.score_samples(X_test)
        print('ROC-AUC:', roc_auc_score(y_test, scores))
    except Exception:
        pass
def eval_ae(cfg):
    X_test = pd.read_csv(f"{cfg['paths']['processed']}/X_test.csv").values
    y_test = pd.read_csv(f"{cfg['paths']['processed']}/y_test.csv")['is_anomaly'].values
    model = keras.models.load_model(cfg['models']['autoencoder']['model_path'])
    recon = model.predict(X_test, verbose=0)
    mse = np.mean((X_test - recon)**2, axis=1)
    thr = np.percentile(mse[y_test==0], 95)
    y_pred = (mse > thr).astype(int)
    print('Autoencoder results')
    print(classification_report(y_test, y_pred, digits=4))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
def main():
    cfg = load_config(); setup_logging(cfg['paths']['log_file'])
    eval_iforest(cfg)
    try:
        eval_ae(cfg)
    except Exception as e:
        print('Autoencoder evaluation skipped:', e)
if __name__ == '__main__':
    main()
