"""
UPDRS regression model training — Zone B.

Trains an SVR/RandomForest on the UCI Parkinson's Telemonitoring Dataset
(https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).

Fetches dataset automatically via ucimlrepo; drops rows with any NaN.

Usage:
    python -m ml.train_model --out ml/model.pkl
    python -m ml.train_model --out ml/model.pkl --model-type svr
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from ucimlrepo import fetch_ucirepo

TARGET_COLUMN = "total_UPDRS"

FEATURE_COLUMNS = [
    "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq5",
    "hnr", "rpde", "dfa", "ppe",
]

# UCI dataset column names (before mapping to FeatureVector fields)
# These are the canonical UCI column names; may have slight variations in real data
UCI_COLUMNS_MAP = {
    "Jitter(%)": "jitter_local",
    "Jitter(%": "jitter_local",
    "Jitter(Abs": "jitter_rap",
    "Jitter(Abs)": "jitter_rap",
    "Shimmer": "shimmer_local",
    "Shimmer(dB": "shimmer_apq5",
    "Shimmer(dB)": "shimmer_apq5",
    "HNR": "hnr",
    "RPDE": "rpde",
    "DFA": "dfa",
    "PPE": "ppe",
}


def load_uci_dataset() -> pd.DataFrame:
    """Fetch UCI Parkinson's Telemonitoring dataset (ID 189) and rename
    columns to match db.contracts.FeatureVector.

    Falls back to loading from data/parkinsons_updrs.csv if UCI fetch fails
    (e.g., SSL certificate issues in isolated environments).
    """
    csv_path = Path("data/parkinsons_updrs.csv")

    # Try UCI fetch first
    try:
        parkinsons = fetch_ucirepo(id=189)
        df = parkinsons.data.features.copy()
        df[TARGET_COLUMN] = parkinsons.data.targets[TARGET_COLUMN]
    except Exception as e:
        # Fallback: load from local CSV if it exists
        if csv_path.exists():
            print(f"⚠️  UCI fetch failed ({type(e).__name__}), loading from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"Could not fetch UCI dataset, and no local {csv_path} found. "
                f"Either resolve network access or download the dataset manually."
            ) from e

    # Map UCI column names to FeatureVector field names
    # Only rename columns that are in the mapping (handles variations in column naming)
    cols_to_rename = {col: UCI_COLUMNS_MAP[col] for col in df.columns if col in UCI_COLUMNS_MAP}
    df = df.rename(columns=cols_to_rename)

    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    dropped = initial_rows - final_rows

    if dropped > 0:
        pct = 100 * dropped / initial_rows
        if pct > 10:
            print(f"⚠️  Dropped {dropped}/{initial_rows} rows ({pct:.1f}%) due to missing values", file=sys.stderr)
        else:
            print(f"ℹ️  Dropped {dropped}/{initial_rows} rows ({pct:.1f}%) due to missing values")

    return df


def train(df: pd.DataFrame, model_type: str = "random_forest"):
    """Train/test split and fit the UPDRS regression model.

    Args:
        df: DataFrame with FEATURE_COLUMNS + TARGET_COLUMN
        model_type: 'svr' or 'random_forest'

    Returns:
        Fitted model (sklearn estimator)
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "svr":
        model = SVR(kernel="rbf", C=100, gamma="scale")
    else:  # random_forest
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Model: {model_type}")
    print(f"Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UPDRS regression model.")
    parser.add_argument("--out", type=str, default="ml/model.pkl")
    parser.add_argument("--model-type", choices=["svr", "random_forest"], default="random_forest")
    args = parser.parse_args()

    print("Fetching UCI Parkinson's Telemonitoring dataset...")
    df = load_uci_dataset()
    print(f"Loaded {len(df)} samples with {len(FEATURE_COLUMNS)} features\n")

    model = train(df, model_type=args.model_type)

    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved model to {args.out}")


if __name__ == "__main__":
    main()
