import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import preprocess

def train_model(X_train, contamination):
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,  # match actual anomaly rate
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    return model

def save_model(model, path='models/anomaly_model.pkl'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, y_train, df, features = preprocess('data/raw/train_FD001.txt')

    # Use actual anomaly rate as contamination
    actual_contamination = round(float(y_train.mean()), 2)
    print(f"\nUsing contamination: {actual_contamination}")

    print("Training Isolation Forest model...")
    model = train_model(X_train, contamination=actual_contamination)

    # Predict: -1 = anomaly, 1 = normal → convert to 0/1
    y_pred = model.predict(X_train)
    y_pred_binary = (y_pred == -1).astype(int)

    print("\n--- Training Results ---")
    print(classification_report(y_train, y_pred_binary,
                                target_names=['Normal', 'Anomaly'],
                                zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_pred_binary))

    save_model(model)
    print("\nDone!")