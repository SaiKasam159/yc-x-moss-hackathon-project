"""
UPDRS prediction — Zone B.

Loads the trained model and scores a FeatureVector.

Read this before trusting the number it returns. Measured with training and
test sets containing *different patients* (5-fold grouped CV on the UCI
telemonitoring data, 42 patients):

    always predict the average    R² = -0.06    MAE = 8.8 UPDRS points
    8 voice features, RF          R² = -0.21    MAE = 9.4
    8 voice features, SVR         R² = -0.23    MAE = 9.5

A negative R² means worse than guessing the group average, so for a patient
the model has not heard before, this score carries no demonstrated
information. The earlier "test R² = 0.35" came from splitting rows at random:
each person contributes ~140 recordings, so the same people appeared on both
sides and the model scored well by recognising individuals. Patient averages
span ~10.4 UPDRS points while one patient varies by only ~2.3, so identity is
most of the signal.

The prediction is therefore reported as experimental, and `confidence_band`
says so in text that reaches the report. What the pipeline measures directly
— jitter, shimmer, HNR, speech rate, pauses — is real and worth tracking per
patient over time; the modelled score is the part that isn't validated.

`anomaly_flag` is not the model's confidence (it has none worth reporting).
It marks a call whose features fall outside the range the model was trained
on, where even its unvalidated output is extrapolation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from db.contracts import FeatureVector, UPDRSPrediction

logger = logging.getLogger(__name__)

MODEL_PATH = "ml/model.pkl"

# Order the model expects. Kept here so an older bare-estimator pickle (no
# bundle metadata) still works.
DEFAULT_FEATURE_COLUMNS = [
    "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq5",
    "hnr", "rpde", "dfa", "ppe",
]

_bundle: Optional[dict] = None


def _load_bundle() -> dict:
    """Load the model. Accepts both the bundle saved by ml/train_model.py and
    a bare estimator from an older run."""
    global _bundle
    if _bundle is None:
        path = Path(MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first: python -m ml.train_model"
            )
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict) and "model" in loaded:
            _bundle = loaded
        else:
            _bundle = {
                "model": loaded,
                "feature_columns": DEFAULT_FEATURE_COLUMNS,
                "feature_ranges": {},
                "metrics": {},
            }
    return _bundle


def out_of_range_features(features: FeatureVector, bundle: Optional[dict] = None) -> list[str]:
    """Features outside the range the model was trained on.

    Anything here means the score is extrapolation — the model has never seen
    input like this, so even its unvalidated output doesn't apply.
    """
    bundle = bundle or _load_bundle()
    outside = []
    for name, (low, high) in (bundle.get("feature_ranges") or {}).items():
        value = getattr(features, name, None)
        if value is None:
            continue
        if value < low or value > high:
            outside.append(f"{name}={value:.4g} (trained on {low:.4g}–{high:.4g})")
    return outside


def _severity(score: float) -> str:
    """Rough UPDRS severity band. Deterministic thresholds, no model involved."""
    if score <= 15:
        return "low"
    if score <= 30:
        return "moderate"
    return "high"


def predict_updrs(features: FeatureVector) -> UPDRSPrediction:
    """Predict a UPDRS score from extracted features.

    The returned score is experimental — see the module docstring. Callers
    should present it alongside that caveat, not as a measurement.
    """
    bundle = _load_bundle()
    model = bundle["model"]
    columns = bundle.get("feature_columns") or DEFAULT_FEATURE_COLUMNS

    import pandas as pd
    row = pd.DataFrame([[getattr(features, c) for c in columns]], columns=columns)
    score = round(float(model.predict(row)[0]), 1)

    outside = out_of_range_features(features, bundle)
    if outside:
        logger.warning(
            "call %s: features outside the model's training range — %s",
            features.call_id, "; ".join(outside),
        )

    test_r2 = (bundle.get("metrics") or {}).get("test_r2")
    validity = (
        f"unvalidated for new patients (held-out R²={test_r2:.2f})"
        if isinstance(test_r2, (int, float)) else "unvalidated for new patients"
    )
    band = f"{_severity(score)} severity, {validity}"
    if outside:
        band += "; input outside training range"

    return UPDRSPrediction(
        call_id=features.call_id,
        predicted_score=score,
        confidence_band=band,
        anomaly_flag=bool(outside),
    )
