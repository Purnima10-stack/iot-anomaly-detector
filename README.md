# 🔧 IoT Anomaly Detector

End-to-end anomaly detection system on NASA Turbofan Engine sensor data.

## What it does
- Ingests real industrial IoT sensor data (NASA CMAPSS dataset)
- Trains an Isolation Forest model to detect anomalous engine behaviour
- Exposes predictions via a lightweight REST API
- Visualizes real-time anomaly scores on a live dashboard

## Architecture
```
Sensor Data → Preprocessing → Isolation Forest Model → REST API → Live Dashboard
```

## Model Performance
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 91% | 73% |
| Precision | 70% | 27% |
| Recall | 70% | 7% |
| F1 Score | 70% | 11% |

> Test performance is lower due to incomplete engine cycles in the test set —
> a known characteristic of the CMAPSS dataset.

## Tech Stack
- **ML Model:** Scikit-learn Isolation Forest
- **Data:** NASA CMAPSS Turbofan Engine Dataset
- **API:** Python HTTP Server
- **Dashboard:** HTML + Chart.js
- **Language:** Python

## How to Run

1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Download dataset from [Kaggle](https://kaggle.com/datasets/behrad3d/nasa-cmaps)
   and place in `data/raw/`
6. Preprocess: `python src/preprocess.py`
7. Train: `python src/train.py`
8. Evaluate: `python src/evaluate.py`
9. Start API: `python api/app.py`
10. Open `dashboard/index.html` in browser

## Dataset
NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS)
- 20,631 training samples
- 21 sensor readings per sample
- Anomaly = engine within last 30 cycles before failure
