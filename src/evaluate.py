import numpy as np
import joblib
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import preprocess

from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score)

def evaluate():
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/anomaly_model.pkl')

    print("Loading and preprocessing test data...")
    X_test, y_test, df, features = preprocess(
        'data/raw/test_FD001.txt',
        scaler=scaler,
        fit_scaler=False
    )

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred == -1).astype(int)
    scores = model.decision_function(X_test)

    print("\n--- Test Evaluation Results ---")
    print(classification_report(y_test, y_pred_binary,
                                target_names=['Normal', 'Anomaly'],
                                zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))

    metrics = {
        "precision": round(precision_score(y_test, y_pred_binary, zero_division=0), 3),
        "recall": round(recall_score(y_test, y_pred_binary, zero_division=0), 3),
        "f1_score": round(f1_score(y_test, y_pred_binary, zero_division=0), 3),
        "accuracy": round(float((y_test == y_pred_binary).mean()), 3),
        "anomaly_rate_actual": round(float(y_test.mean()), 3),
        "anomaly_rate_predicted": round(float(y_pred_binary.mean()), 3),
        "total_samples": int(len(y_test)),
        "anomalies_detected": int(y_pred_binary.sum())
    }

    os.makedirs('models', exist_ok=True)
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Saved Metrics ---")
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    evaluate()