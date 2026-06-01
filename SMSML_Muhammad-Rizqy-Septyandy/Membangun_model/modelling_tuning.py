"""Advanced model training: MLflow Tracking ke DagsHub (online).

- Hyperparameter tuning via GridSearchCV
- Manual logging (autolog dimatikan) dengan metriks tambahan di luar autolog
- Minimal 2 artefak tambahan: confusion_matrix.png, roc_curve.png,
  feature_importance.csv, classification_report.json

Sebelum menjalankan, set environment variables (jangan commit credentials):
    export MLFLOW_TRACKING_URI=https://dagshub.com/septyandy08/SMSML-Personality.mlflow
    export MLFLOW_TRACKING_USERNAME=septyandy08
    export MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>

Atau gunakan dagshub.init() seperti dokumentasi:
    import dagshub
    dagshub.init(repo_owner='septyandy08', repo_name='SMSML-Personality', mlflow=True)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV

DAGSHUB_REPO_OWNER = "septyandy08"
DAGSHUB_REPO_NAME = "SMSML-Personality"


def init_tracking(use_dagshub: bool) -> None:
    if use_dagshub:
        try:
            import dagshub

            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] dagshub.init gagal ({exc}); fallback ke MLFLOW_TRACKING_URI env.")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Personality-Classification-Advance")


def load_split(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    X_train = train.drop(columns=["Personality"])
    y_train = train["Personality"]
    X_test = test.drop(columns=["Personality"])
    y_test = test["Personality"]
    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Extrovert", "Introvert"])
    ax.set_yticklabels(["Extrovert", "Introvert"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(y_true, y_proba, path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="personality_preprocessing")
    parser.add_argument(
        "--use_dagshub",
        action="store_true",
        help="Aktifkan untuk mengirim tracking ke DagsHub (default: lokal mlruns/).",
    )
    args = parser.parse_args()

    init_tracking(use_dagshub=args.use_dagshub)
    X_train, y_train, X_test, y_test = load_split(Path(args.data_dir))

    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True,
    )

    with mlflow.start_run(run_name="rf_tuning_dagshub"):
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("model_family", "RandomForestClassifier")

        preds = best.predict(X_test)
        proba = best.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
            "log_loss": log_loss(y_test, proba),
            "mcc": matthews_corrcoef(y_test, preds),
            "best_cv_f1": grid.best_score_,
            "train_test_gap": grid.cv_results_["mean_train_score"][grid.best_index_]
            - grid.cv_results_["mean_test_score"][grid.best_index_],
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        cm = confusion_matrix(y_test, preds)
        cm_path = artifacts_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, cm_path)
        mlflow.log_artifact(str(cm_path))

        roc_path = artifacts_dir / "roc_curve.png"
        plot_roc(y_test, proba, roc_path)
        mlflow.log_artifact(str(roc_path))

        fi = pd.DataFrame(
            {"feature": X_train.columns, "importance": best.feature_importances_}
        ).sort_values("importance", ascending=False)
        fi_path = artifacts_dir / "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        report = classification_report(y_test, preds, output_dict=True)
        report_path = artifacts_dir / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        model_path = artifacts_dir / "best_model.joblib"
        joblib.dump(best, model_path)
        mlflow.log_artifact(str(model_path))

        mlflow.sklearn.log_model(best, artifact_path="model")

        print("Best params:", grid.best_params_)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
