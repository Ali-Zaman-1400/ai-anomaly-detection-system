import numpy as np, pandas as pd
from sklearn.datasets import make_blobs
from utils import load_config, ensure_dir, setup_logging
def make_synthetic(n_samples=5000, n_features=6, contamination=0.03, random_state=42):
    X, _ = make_blobs(n_samples=int(n_samples*(1-contamination)), n_features=n_features,
                      centers=3, cluster_std=1.2, random_state=random_state)
    rng = np.random.RandomState(random_state)
    n_anom = int(n_samples * contamination)
    anomalies = rng.uniform(low=8, high=15, size=(n_anom, n_features))
    X_all = np.vstack([X, anomalies])
    y_all = np.hstack([np.zeros(len(X)), np.ones(n_anom)]).astype(int)
    return X_all, y_all
def main():
    cfg = load_config()
    setup_logging(cfg["paths"]["log_file"])
    raw_dir = cfg["paths"]["raw"]
    ensure_dir(raw_dir)
    X, y = make_synthetic(**cfg["data"])
    df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])
    df["is_anomaly"] = y
    out_csv = f"{raw_dir}/synthetic_transactions.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved raw data to {out_csv}")
if __name__ == "__main__":
    main()
