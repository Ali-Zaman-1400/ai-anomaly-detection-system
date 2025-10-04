import pandas as pd, numpy as np, joblib
from tensorflow import keras
from tensorflow.keras import layers
from utils import load_config, setup_logging
def build_autoencoder(input_dim:int):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    bottleneck = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(bottleneck)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model
def main():
    cfg = load_config(); setup_logging(cfg['paths']['log_file'])
    X_train = pd.read_csv(f"{cfg['paths']['processed']}/X_train.csv").values
    y_train = pd.read_csv(f"{cfg['paths']['processed']}/y_train.csv")['is_anomaly'].values
    Xn = X_train[y_train==0]
    params = cfg['models']['autoencoder']
    model = build_autoencoder(Xn.shape[1])
    model.fit(Xn, Xn, epochs=params['epochs'], batch_size=params['batch_size'],
              validation_split=params['validation_split'], verbose=1)
    model.save(params['model_path'])
    print(f"Autoencoder saved to {params['model_path']}")
if __name__ == '__main__':
    main()
