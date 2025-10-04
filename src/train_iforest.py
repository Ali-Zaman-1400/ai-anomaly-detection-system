import pandas as pd, joblib
from sklearn.ensemble import IsolationForest
from utils import load_config, setup_logging
def main():
    cfg = load_config(); setup_logging(cfg['paths']['log_file'])
    X_train = pd.read_csv(f"{cfg['paths']['processed']}/X_train.csv").values
    params = cfg['models']['iforest']
    model = IsolationForest(n_estimators=params['n_estimators'],
                            max_samples=params['max_samples'],
                            contamination=params['contamination'],
                            random_state=params['random_state'])
    model.fit(X_train)
    joblib.dump(model, params['model_path'])
    print(f"Isolation Forest saved to {params['model_path']}")
if __name__ == "__main__":
    main()
