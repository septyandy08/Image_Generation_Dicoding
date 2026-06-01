"""Client inference sederhana untuk model MLflow serve.

Mengirim request ke endpoint yang sudah di-serve (default :5001) lalu
mencetak hasil prediksi. Dipakai untuk membangkitkan trafik agar Prometheus
exporter (port 8000) menghasilkan metriks yang bisa di-visualisasi Grafana.

Usage:
    # Terminal 1: serve model
    mlflow models serve -m runs:/<RUN_ID>/model -h 127.0.0.1 -p 5001 --no-conda

    # Terminal 2: jalankan exporter
    python 3.prometheus_exporter.py

    # Terminal 3: generate trafik
    python 7.inference.py --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import time

import requests

ENDPOINT_DEFAULT = "http://127.0.0.1:5001/invocations"
COLUMNS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]


def make_row() -> list[float]:
    return [
        random.uniform(-2.0, 2.0),
        random.choice([0, 1]),
        random.uniform(-2.0, 2.0),
        random.uniform(-2.0, 2.0),
        random.choice([0, 1]),
        random.uniform(-2.0, 2.0),
        random.uniform(-2.0, 2.0),
    ]


def predict(endpoint: str, n: int, sleep: float) -> None:
    label_map = {0: "Extrovert", 1: "Introvert"}
    for i in range(n):
        body = {"dataframe_split": {"columns": COLUMNS, "data": [make_row()]}}
        try:
            r = requests.post(endpoint, json=body, timeout=5)
            if r.status_code == 200:
                preds = r.json().get("predictions", r.json())
                cls = int(preds[0]) if isinstance(preds, list) else int(preds)
                print(f"[{i+1:03d}] -> {label_map.get(cls, cls)}")
            else:
                print(f"[{i+1:03d}] HTTP {r.status_code}: {r.text[:120]}")
        except requests.RequestException as exc:
            print(f"[{i+1:03d}] ERROR {exc}")
        time.sleep(sleep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=ENDPOINT_DEFAULT)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()
    predict(args.endpoint, args.n, args.sleep)


if __name__ == "__main__":
    main()
