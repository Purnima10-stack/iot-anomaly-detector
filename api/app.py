from http.server import HTTPServer, BaseHTTPRequestHandler
import joblib
import numpy as np
import json
import os

# Load model and scaler
model = joblib.load(os.path.join('models', 'anomaly_model.pkl'))
scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress request logs to save memory

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_json({"status": "ok", "message": "IoT Anomaly Detector running"})

        elif self.path == '/metrics':
            try:
                with open(os.path.join('models', 'metrics.json')) as f:
                    self.send_json(json.load(f))
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length))

        if self.path == '/predict':
            try:
                sensors = np.array(body['sensors']).reshape(1, -1)
                if sensors.shape[1] != 19:
                    self.send_json({"error": f"Expected 19 values, got {sensors.shape[1]}"}, 400)
                    return
                scaled = scaler.transform(sensors)
                pred = model.predict(scaled)[0]
                score = float(model.decision_function(scaled)[0])
                is_anomaly = pred == -1
                self.send_json({
                    "is_anomaly": bool(is_anomaly),
                    "anomaly_score": round(score, 4),
                    "status": "ANOMALY DETECTED" if is_anomaly else "Normal",
                    "confidence": round(abs(score), 4)
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif self.path == '/predict/batch':
            try:
                readings = np.array(body['readings'])
                scaled = scaler.transform(readings)
                predictions = model.predict(scaled)
                scores = model.decision_function(scaled)
                results = [{
                    "is_anomaly": bool(p == -1),
                    "anomaly_score": round(float(s), 4),
                    "status": "ANOMALY DETECTED" if p == -1 else "Normal"
                } for p, s in zip(predictions, scores)]
                self.send_json({
                    "results": results,
                    "total": len(results),
                    "anomalies_found": sum(1 for r in results if r['is_anomaly'])
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_json({"error": "Not found"}, 404)

if __name__ == '__main__':
    print("Starting IoT Anomaly Detector API on port 5000...")
    HTTPServer(('', 5000), Handler).serve_forever()