"""Training entrypoint untuk MLflow Project CI.

Mempelajari RandomForest pada dataset personality_preprocessing dan
menyimpan model + metrik via MLflow. Dipakai sebagai entrypoint pada
file `MLProject` agar bisa dipanggil `mlflow run`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_split(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    X_train = train.drop(columns=["Personality"])
    y_train = train["Personality"]
    X_test = test.drop(columns=["Personality"])
    y_test = test["Personality"]
    return X_train, y_train, X_test, y_test


def save_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="personality_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_split(Path(args.data_dir))

    if os.environ.get("MLFLOW_RUN_ID") is None:
        mlflow.set_experiment("Personality-CI")

    with mlflow.start_run(run_name="ci_rf_run"):
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("model_family", "RandomForestClassifier")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        out = Path("artifacts")
        out.mkdir(exist_ok=True)

        cm = confusion_matrix(y_test, preds)
        save_confusion(cm, out / "confusion_matrix.png")
        mlflow.log_artifact(str(out / "confusion_matrix.png"))

        report = classification_report(y_test, preds, output_dict=True)
        (out / "classification_report.json").write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(out / "classification_report.json"))

        joblib.dump(model, out / "model.joblib")
        mlflow.log_artifact(str(out / "model.joblib"))

        mlflow.sklearn.log_model(model, artifact_path="model")

        print("CI training metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
