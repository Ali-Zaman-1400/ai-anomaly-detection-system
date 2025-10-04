import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from utils import load_config, ensure_dir, setup_logging
def main():
    cfg = load_config()
    setup_logging(cfg["paths"]["log_file"])
    raw = f"{cfg['paths']['raw']}/synthetic_transactions.csv"
    df = pd.read_csv(raw)
    X = df.drop(columns=['is_anomaly']).values
    y = df['is_anomaly'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    proc = cfg['paths']['processed']; ensure_dir(proc)
    pd.DataFrame(X_train).to_csv(f"{proc}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{proc}/X_test.csv", index=False)
    pd.DataFrame(y_train, columns=['is_anomaly']).to_csv(f"{proc}/y_train.csv", index=False)
    pd.DataFrame(y_test, columns=['is_anomaly']).to_csv(f"{proc}/y_test.csv", index=False)
    ensure_dir(cfg['paths']['models'])
    joblib.dump(scaler, f"{cfg['paths']['models']}/scaler.joblib")
    print("Preprocessing completed.")
if __name__ == "__main__":
    main()
