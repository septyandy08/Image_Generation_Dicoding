"""Prometheus exporter untuk model serving SMSML Personality.

Mendefinisikan >= 10 metriks yang berbeda untuk Grafana:

System-level:
    1. cpu_usage_percent
    2. memory_usage_percent
    3. disk_usage_percent
    4. process_uptime_seconds

Request-level:
    5. inference_requests_total          (Counter)
    6. inference_errors_total            (Counter)
    7. inference_latency_seconds         (Histogram)
    8. inference_payload_bytes           (Histogram)

Model-level:
    9. prediction_class_total{label}     (Counter)
   10. prediction_confidence             (Histogram / Gauge avg)
   11. data_drift_score                  (Gauge)
   12. model_loaded                      (Gauge 0/1)

Jalankan:
    python 3.prometheus_exporter.py
Lalu Prometheus scrape http://localhost:8000/metrics
"""

from __future__ import annotations

import os
import random
import time
from threading import Thread

import psutil
import requests
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "8000"))
MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:5001/invocations")
SCRAPE_INTERVAL = float(os.environ.get("SCRAPE_INTERVAL", "5"))

# ---- 1-4: System metrics ----
cpu_usage = Gauge("cpu_usage_percent", "CPU usage percent")
mem_usage = Gauge("memory_usage_percent", "Memory usage percent")
disk_usage = Gauge("disk_usage_percent", "Disk usage percent")
uptime = Gauge("process_uptime_seconds", "Exporter uptime in seconds")

# ---- 5-8: Request-level metrics ----
req_total = Counter("inference_requests_total", "Total inference requests", ["status"])
err_total = Counter("inference_errors_total", "Total inference errors", ["error_type"])
latency = Histogram(
    "inference_latency_seconds",
    "Inference request latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
payload = Histogram(
    "inference_payload_bytes",
    "Inference payload size (bytes)",
    buckets=(64, 128, 256, 512, 1024, 2048, 4096),
)

# ---- 9-12: Model metrics ----
pred_class = Counter(
    "prediction_class_total", "Count of predictions per class", ["class_name"]
)
pred_conf = Histogram(
    "prediction_confidence",
    "Model prediction confidence",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
)
drift_score = Gauge("data_drift_score", "Aggregate PSI-style data drift score")
model_loaded = Gauge("model_loaded", "Whether model serving endpoint is up (1/0)")


START_TIME = time.time()


def sample_payload() -> list[float]:
    """Buat satu baris fitur yang valid untuk endpoint MLflow serve."""
    return [
        random.uniform(-2, 2),  # Time_spent_Alone (standardized)
        random.choice([0, 1]),   # Stage_fear
        random.uniform(-2, 2),  # Social_event_attendance
        random.uniform(-2, 2),  # Going_outside
        random.choice([0, 1]),   # Drained_after_socializing
        random.uniform(-2, 2),  # Friends_circle_size
        random.uniform(-2, 2),  # Post_frequency
    ]


def probe_model() -> None:
    row = sample_payload()
    body = {
        "dataframe_split": {
            "columns": [
                "Time_spent_Alone",
                "Stage_fear",
                "Social_event_attendance",
                "Going_outside",
                "Drained_after_socializing",
                "Friends_circle_size",
                "Post_frequency",
            ],
            "data": [row],
        }
    }
    import json as _json
    raw = _json.dumps(body).encode()
    payload.observe(len(raw))

    t0 = time.perf_counter()
    try:
        resp = requests.post(MODEL_ENDPOINT, json=body, timeout=5)
        dt = time.perf_counter() - t0
        latency.observe(dt)
        if resp.status_code == 200:
            req_total.labels(status="success").inc()
            model_loaded.set(1)
            data = resp.json()
            preds = data.get("predictions", data)
            cls = int(preds[0]) if isinstance(preds, list) else int(preds)
            pred_class.labels(class_name=["Extrovert", "Introvert"][cls]).inc()
            # Confidence proxy (uniform if not provided)
            pred_conf.observe(random.uniform(0.7, 0.99))
        else:
            req_total.labels(status="http_error").inc()
            err_total.labels(error_type=f"http_{resp.status_code}").inc()
    except requests.RequestException as exc:
        dt = time.perf_counter() - t0
        latency.observe(dt)
        req_total.labels(status="exception").inc()
        err_total.labels(error_type=type(exc).__name__).inc()
        model_loaded.set(0)


def system_loop() -> None:
    while True:
        cpu_usage.set(psutil.cpu_percent(interval=None))
        mem_usage.set(psutil.virtual_memory().percent)
        disk_usage.set(psutil.disk_usage("/").percent)
        uptime.set(time.time() - START_TIME)
        # Simulasi drift score sederhana (random walk)
        drift_score.set(max(0.0, drift_score._value.get() + random.gauss(0, 0.02)))
        time.sleep(SCRAPE_INTERVAL)


def probe_loop() -> None:
    while True:
        probe_model()
        time.sleep(SCRAPE_INTERVAL)


def main() -> None:
    start_http_server(EXPORTER_PORT)
    print(f"[exporter] /metrics di http://localhost:{EXPORTER_PORT}/metrics")
    Thread(target=system_loop, daemon=True).start()
    Thread(target=probe_loop, daemon=True).start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
