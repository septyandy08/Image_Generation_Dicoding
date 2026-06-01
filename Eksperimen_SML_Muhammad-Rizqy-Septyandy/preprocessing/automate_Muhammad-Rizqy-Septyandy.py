"""Automated preprocessing pipeline untuk Personality Classification.

Revisi sesuai feedback reviewer:
- Split train/test SEBELUM fit transformer (mencegah data leakage).
- `fit_transform` hanya pada `X_train`, `transform` saja pada `X_test`.
- Imputer + scaler dibungkus `Pipeline` + `ColumnTransformer` agar bisa
  dipakai langsung saat deployment tanpa memuat scaler terpisah.

Usage:
    python automate_Muhammad-Rizqy-Septyandy.py \
        --input ../personality_raw/personality_dataset.csv \
        --output ./personality_preprocessing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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
    return pd.read_csv(path)


def clean_and_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Operasi deterministik yang aman dilakukan sebelum split."""
    df = df.drop_duplicates().reset_index(drop=True)

    df[CAT_COLS] = df[CAT_COLS].fillna(df[CAT_COLS].mode().iloc[0])
    for c in CAT_COLS:
        df[c] = df[c].map({"Yes": 1, "No": 0}).astype(int)

    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])
    return df, le


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUM_COLS),
            ("cat", categorical_pipeline, CAT_COLS),
        ]
    )


def preprocess(input_path: str, output_dir: str, test_size: float = 0.2, seed: int = 42) -> dict:
    out_dir = Path(output_dir)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    n_raw = len(df)
    df, le = clean_and_encode(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    preprocessor = build_preprocessor()
    X_train_arr = preprocessor.fit_transform(X_train)   # fit hanya di train
    X_test_arr = preprocessor.transform(X_test)         # test hanya transform

    feature_order = NUM_COLS + CAT_COLS
    X_train_prep = pd.DataFrame(X_train_arr, columns=feature_order, index=X_train.index)
    X_test_prep = pd.DataFrame(X_test_arr, columns=feature_order, index=X_test.index)

    X_train_prep.assign(Personality=y_train.values).to_csv(out_dir / "train.csv", index=False)
    X_test_prep.assign(Personality=y_test.values).to_csv(out_dir / "test.csv", index=False)

    full = pd.concat(
        [
            X_train_prep.assign(Personality=y_train.values),
            X_test_prep.assign(Personality=y_test.values),
        ],
        ignore_index=True,
    )
    full.to_csv(out_dir / "personality_preprocessed.csv", index=False)

    joblib.dump(preprocessor, art_dir / "preprocessor.joblib")
    joblib.dump(le, art_dir / "label_encoder.joblib")

    return {
        "rows_raw": n_raw,
        "rows_clean": len(df),
        "train_size": len(X_train_prep),
        "test_size": len(X_test_prep),
        "n_features": len(feature_order),
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
