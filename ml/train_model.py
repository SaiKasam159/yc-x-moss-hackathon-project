"""
UPDRS regression model training — Zone B.

Trains an SVR/RandomForest on the UCI Parkinson's Telemonitoring Dataset
(https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).
Not yet run — dataset isn't downloaded (see /data). This is a skeleton
script with the pipeline shape; fill in once data is in place.

Usage (once implemented):
    python -m ml.train_model --data data/parkinsons_updrs.csv --out ml/model.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# UCI dataset target column is total_UPDRS (or motor_UPDRS — TODO(ml): confirm
# which target this project is predicting against and update FeatureVector /
# db.schema.sql's confidence_band logic accordingly if needed).
TARGET_COLUMN = "total_UPDRS"

FEATURE_COLUMNS = [
    "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq5",
    "hnr", "rpde", "dfa", "ppe",
]


def load_uci_dataset(path: str) -> pd.DataFrame:
    """TODO(ml): implement — load + rename UCI columns to match
    db.contracts.FeatureVector field names."""
    raise NotImplementedError("load_uci_dataset is a stub — implement once data/ is populated.")


def train(df: pd.DataFrame, model_type: str = "random_forest"):
    """TODO(ml): implement train/test split + fit. model_type: 'svr' | 'random_forest'."""
    raise NotImplementedError("train is a stub — implement.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UPDRS regression model.")
    parser.add_argument("--data", type=str, default="data/parkinsons_updrs.csv")
    parser.add_argument("--out", type=str, default="ml/model.pkl")
    parser.add_argument("--model-type", choices=["svr", "random_forest"], default="random_forest")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset not found at {args.data}. Download the UCI Parkinson's "
            f"Telemonitoring dataset into /data first (see README)."
        )

    df = load_uci_dataset(args.data)
    model = train(df, model_type=args.model_type)

    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
