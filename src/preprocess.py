import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath):
    cols = ['unit_id', 'cycle', 'op1', 'op2', 'op3'] + \
           [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=cols, engine='python')
    df = df.dropna(axis=1)
    return df

def add_rul(df):
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def label_anomalies(df, rul_threshold=30):
    df['anomaly'] = (df['RUL'] <= rul_threshold).astype(int)
    return df

def get_feature_columns(df):
    drop_cols = ['unit_id', 'cycle', 'RUL', 'anomaly', 'op1', 'op2', 'op3']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols

def preprocess(filepath, scaler=None, fit_scaler=True):
    df = load_data(filepath)
    df = add_rul(df)
    df = label_anomalies(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df['anomaly'].values

    print(f"  Normal samples: {(y==0).sum()} | Anomaly samples: {(y==1).sum()}")
    print(f"  Anomaly rate: {y.mean():.2%}")

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, df, feature_cols

if __name__ == "__main__":
    X, y, df, features = preprocess('data/raw/train_FD001.txt')
    print(f"Data shape: {X.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Features: {features}")