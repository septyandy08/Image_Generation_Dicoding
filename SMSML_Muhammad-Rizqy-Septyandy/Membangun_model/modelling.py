"""Basic model training dengan MLflow autolog (lokal)."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_split(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    X_train = train.drop(columns=["Personality"])
    y_train = train["Personality"]
    X_test = test.drop(columns=["Personality"])
    y_test = test["Personality"]
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="personality_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_split(Path(args.data_dir))

    mlflow.set_experiment("Personality-Classification-Basic")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="rf_basic_autolog"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
