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















What Happened in This Project
The Big Picture
You built a system that monitors industrial engine sensors and automatically flags when an engine is about to fail. This is exactly what companies like Siemens and GE do in real life for predictive maintenance — you just built a simplified version of it using real NASA data.

The Data
You used the NASA CMAPSS Turbofan Engine Dataset — real sensor readings from aircraft engines running until they fail. Each row in the dataset represents one engine at one point in time, with 19 sensor readings (temperature, pressure, speed, etc.) recorded every cycle.
The key insight: engines degrade gradually before failing. So the last 30 cycles before failure are labeled as "anomalous" — the engine is still running but showing signs of wear.

What Each File Does
src/preprocess.py
This is the data preparation file. It does 4 things:

Loads the raw data — reads the text file and assigns proper column names to all 26 columns
Calculates RUL (Remaining Useful Life) — for each engine, figures out how many cycles it has left before failure. If an engine ran for 200 cycles total and you're at cycle 150, RUL = 50
Labels anomalies — marks the last 30 cycles of each engine as anomalous (RUL ≤ 30 = anomaly). This is how the model knows what "bad" looks like
Scales the features — converts all sensor values to the same scale using StandardScaler, so no single sensor dominates the model. Saves the scaler to models/scaler.pkl for reuse


src/train.py
This is the model training file. It:

Calls preprocess.py to get clean, scaled data
Trains an Isolation Forest model — an unsupervised algorithm that learns what "normal" looks like by randomly splitting data. Points that are easy to isolate are anomalies; points that need many splits to isolate are normal
Sets contamination=0.15 (matching your actual 15% anomaly rate) so the model knows roughly how many anomalies to expect
Evaluates on training data and prints precision, recall, F1 score
Saves the trained model to models/anomaly_model.pkl

Why Isolation Forest? It's perfect for IoT anomaly detection because it doesn't need a large labeled dataset — it learns normality on its own, just like a real industrial monitoring system would.

src/evaluate.py
This is the testing file. It:

Loads the saved model and scaler
Runs them on the test dataset (data the model has never seen)
Computes precision, recall, F1, and accuracy
Saves all metrics to models/metrics.json — this JSON file is later read by the dashboard to display live model performance

The test results (73% accuracy) were lower than training (91%) — which is expected because the test file contains incomplete engine runs, so the RUL labeling is less reliable. This is a known characteristic of the CMAPSS dataset and a good thing to explain in interviews.

api/app.py
This is the REST API file. It:

Loads the trained model and scaler into memory once at startup
Listens on port 5000 for incoming HTTP requests
Exposes 3 endpoints:

GET /health — confirms the API is running
POST /predict — accepts 19 sensor values, scales them, runs them through the model, returns whether it's an anomaly plus a confidence score
POST /predict/batch — same but for multiple readings at once
GET /metrics — returns the saved evaluation metrics as JSON



This is what a real IoT monitoring system would use — sensors send readings to the API, the API responds with anomaly status in real time.

dashboard/index.html
This is the visualization layer. It:

On load, calls /metrics to display model performance (precision, recall, F1, accuracy) at the top
Has two manual buttons — "Simulate Normal" sends typical sensor values to the API, "Simulate Anomaly" sends extreme values
Has an "Auto Simulate" button that fires a reading every 1.5 seconds with a 20% chance of anomaly — simulating a live sensor stream
Plots anomaly scores on a live Chart.js line chart — green dots for normal, red dots for anomalies
Shows a live log of every reading with timestamp and result


The Full Flow End to End
Raw sensor data (text file)
        ↓
preprocess.py → cleans, labels, scales data
        ↓
train.py → trains Isolation Forest, saves model
        ↓
evaluate.py → tests on unseen data, saves metrics
        ↓
api/app.py → loads model, serves predictions via HTTP
        ↓
dashboard/index.html → sends sensor readings to API, visualizes results live

