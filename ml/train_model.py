"""
UPDRS regression model training — Zone B.

Trains on the UCI Parkinson's Telemonitoring dataset
(https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring): ~5,875
voice recordings from 42 people with early-stage Parkinson's, each with a
clinician-assessed UPDRS score.

Usage:
    python -m ml.train_model                      # downloads the data if needed
    python -m ml.train_model --model-type svr
    python -m ml.train_model --random-split       # the old, leaky split, for comparison

Three things were wrong here and are fixed:

1. Two columns were mapped to the wrong features. Jitter(Abs) (absolute
   jitter in seconds, median 0.00003) was loaded as jitter_rap, and
   Shimmer(dB) (median 0.253) as shimmer_apq5. At prediction time Praat sends
   RAP jitter (~0.001) and APQ5 shimmer (~0.006), so two of the eight
   features the model relies on arrived ~37x too large and ~40x too small.
   The model was effectively trained on different measurements than it is
   given in production.

2. The split was random over rows. Each person contributes ~140 recordings,
   so a random split puts the same person in train and test, and the model
   can score well by recognising the person rather than the symptoms. The
   split is now by subject, which is the number that says whether this
   generalises to a new patient. --random-split reproduces the old behaviour
   so the difference is visible.

3. The dataset download was broken (ucimlrepo raises DatasetNotFoundError),
   and the file is gitignored, so a fresh clone could not train at all. The
   data is now fetched directly from the UCI archive when missing.
"""

from __future__ import annotations

import argparse
import datetime
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, train_test_split
from sklearn.svm import SVR

TARGET_COLUMN = "total_UPDRS"
SUBJECT_COLUMN = "subject#"

# The order predict.py sends features in; do not reorder without changing it.
FEATURE_COLUMNS = [
    "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq5",
    "hnr", "rpde", "dfa", "ppe",
]

# UCI column -> FeatureVector field. Each maps to the same measurement the
# extractor produces in ml/audio.py (Praat's local/RAP jitter and
# local/APQ5 shimmer, both as fractions).
UCI_COLUMNS_MAP = {
    "Jitter(%)": "jitter_local",
    "Jitter:RAP": "jitter_rap",
    "Shimmer": "shimmer_local",
    "Shimmer:APQ5": "shimmer_apq5",
    "HNR": "hnr",
    "RPDE": "rpde",
    "DFA": "dfa",
    "PPE": "ppe",
}

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/telemonitoring/parkinsons_updrs.data"
)
DEFAULT_DATA_PATH = Path("data/parkinsons_updrs.data")


def download_dataset(path: Path = DEFAULT_DATA_PATH) -> Path:
    """Fetch the dataset from the UCI archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the UCI Parkinson's Telemonitoring dataset to {path} ...")
    urllib.request.urlretrieve(DATA_URL, path)
    print(f"Downloaded {path.stat().st_size / 1024:.0f} KB")
    return path


def load_uci_dataset(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the dataset, downloading it if it isn't there yet, and rename
    columns to the FeatureVector field names."""
    path = Path(data_path)
    if not path.exists():
        download_dataset(path)

    df = pd.read_csv(path)
    missing = [c for c in UCI_COLUMNS_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    df = df.rename(columns=UCI_COLUMNS_MAP)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, SUBJECT_COLUMN])
    if before != len(df):
        print(f"Dropped {before - len(df)} of {before} rows with missing values")
    return df


def _fit(model_type: str):
    if model_type == "svr":
        return SVR(kernel="rbf", C=100, gamma="scale")
    # Depth-limited and leaf-constrained: unconstrained trees memorised the
    # training rows (train R² 0.91 vs test 0.35).
    return RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )


def train(df: pd.DataFrame, model_type: str = "random_forest", group_split: bool = True):
    """Fit the model and report how well it generalises.

    With group_split (the default) train and test contain different people,
    which is the question that matters: does this work on a patient the model
    has never heard? A random row split leaks the same person into both.
    """
    X, y, groups = df[FEATURE_COLUMNS], df[TARGET_COLUMN], df[SUBJECT_COLUMN]

    if group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        split_desc = "held-out patients"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42
        )
        split_desc = "random rows (same patients on both sides — leaky)"

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = _fit(model_type)
    model.fit(X_train, y_train)

    train_r2, test_r2 = model.score(X_train, y_train), model.score(X_test, y_test)
    print(f"\nModel: {model_type}, split: {split_desc}")
    print(f"  train rows {len(X_train)} ({groups.iloc[train_idx].nunique()} patients), "
          f"test rows {len(X_test)} ({groups.iloc[test_idx].nunique()} patients)")
    print(f"  train R² = {train_r2:.3f}")
    print(f"  test  R² = {test_r2:.3f}")
    print(f"  overfitting gap = {train_r2 - test_r2:.3f}")

    if group_split:
        cv = cross_val_score(
            _fit(model_type), X, y, groups=groups,
            cv=GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=0),
        )
        print(f"  5x grouped CV R² = {cv.mean():.3f} (±{cv.std():.3f})")

    metrics = {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "split": split_desc,
        "n_train_subjects": int(groups.iloc[train_idx].nunique()),
        "n_test_subjects": int(groups.iloc[test_idx].nunique()),
    }
    return model, metrics


def build_bundle(model, df: pd.DataFrame, model_type: str, metrics: dict) -> dict:
    """What gets pickled: the model plus what's needed to use it honestly.

    The feature ranges let predict.py tell when a call's features fall outside
    anything the model was trained on, instead of returning a confident number
    for input it has never seen.
    """
    ranges = {
        col: (float(df[col].quantile(0.01)), float(df[col].quantile(0.99)))
        for col in FEATURE_COLUMNS
    }
    return {
        "model": model,
        "model_type": model_type,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_ranges": ranges,
        "target": TARGET_COLUMN,
        "target_range": (float(df[TARGET_COLUMN].min()), float(df[TARGET_COLUMN].max())),
        "metrics": metrics,
        "n_subjects": int(df[SUBJECT_COLUMN].nunique()),
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UPDRS regression model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--out", default="ml/model.pkl")
    parser.add_argument("--model-type", choices=["svr", "random_forest"], default="random_forest")
    parser.add_argument("--random-split", action="store_true",
                        help="split randomly over rows instead of by patient (leaky; for comparison)")
    args = parser.parse_args()

    df = load_uci_dataset(args.data)
    print(f"Loaded {len(df)} recordings from {df[SUBJECT_COLUMN].nunique()} patients, "
          f"{len(FEATURE_COLUMNS)} features")

    model, metrics = train(df, model_type=args.model_type, group_split=not args.random_split)

    bundle = build_bundle(model, df, args.model_type, metrics)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
