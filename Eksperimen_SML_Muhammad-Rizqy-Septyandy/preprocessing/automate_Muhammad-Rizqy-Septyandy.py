"""Automated preprocessing pipeline untuk Personality Classification.

Menjalankan tahapan yang sama dengan notebook Eksperimen_Muhammad-Rizqy-Septyandy.ipynb:
load -> dedup -> impute -> encode -> scale -> split -> save.

Usage:
    python automate_Muhammad-Rizqy-Septyandy.py \
        --input ../personality_raw/personality_dataset.csv \
        --output ./personality_preprocessing
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

NUM_COLS = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency",
]
CAT_COLS = ["Stage_fear", "Drained_after_socializing"]
TARGET = "Personality"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def impute(df: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer, SimpleImputer]:
    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[NUM_COLS] = num_imp.fit_transform(df[NUM_COLS])
    df[CAT_COLS] = cat_imp.fit_transform(df[CAT_COLS])
    return df, num_imp, cat_imp


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    for c in CAT_COLS:
        df[c] = df[c].map({"Yes": 1, "No": 0}).astype(int)
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])
    return df, le


def scale(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
    return df, scaler


def split_and_save(df: pd.DataFrame, out_dir: Path, test_size: float = 0.2, seed: int = 42) -> None:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "personality_preprocessed.csv", index=False)
    X_train.assign(Personality=y_train).to_csv(out_dir / "train.csv", index=False)
    X_test.assign(Personality=y_test).to_csv(out_dir / "test.csv", index=False)


def preprocess(input_path: str, output_dir: str) -> dict:
    out_dir = Path(output_dir)
    df = load_data(input_path)
    n_raw = len(df)
    df = clean(df)
    df, num_imp, cat_imp = impute(df)
    df, le = encode(df)
    df, scaler = scale(df)
    split_and_save(df, out_dir)

    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(num_imp, artifacts_dir / "num_imputer.joblib")
    joblib.dump(cat_imp, artifacts_dir / "cat_imputer.joblib")
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")
    joblib.dump(le, artifacts_dir / "label_encoder.joblib")

    return {
        "rows_raw": n_raw,
        "rows_clean": len(df),
        "n_features": df.shape[1] - 1,
        "class_distribution": df[TARGET].value_counts().to_dict(),
        "output_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../personality_raw/personality_dataset.csv")
    parser.add_argument("--output", default="./personality_preprocessing")
    args = parser.parse_args()

    summary = preprocess(args.input, args.output)
    print("Preprocessing selesai:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
