"""
UPDRS prediction — Zone B.

Loads a trained regression model (sklearn) and predicts UPDRS scores
from extracted acoustic features (FeatureVector).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from db.contracts import FeatureVector, UPDRSPrediction

MODEL_PATH = "ml/model.pkl"

_model = None


def _load_model():
    global _model
    if _model is None:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Train the model first: python -m ml.train_model"
            )
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
    return _model


def predict_updrs(features: FeatureVector) -> UPDRSPrediction:
    """Predict UPDRS score from extracted acoustic features.

    Args:
        features: FeatureVector with all acoustic/prosody measurements

    Returns:
        UPDRSPrediction with predicted score and confidence band
    """
    model = _load_model()

    import pandas as pd
    feature_names = [
        "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq5",
        "hnr", "rpde", "dfa", "ppe",
    ]
    feature_values = [
        features.jitter_local,
        features.jitter_rap,
        features.shimmer_local,
        features.shimmer_apq5,
        features.hnr,
        features.rpde,
        features.dfa,
        features.ppe,
    ]

    feature_df = pd.DataFrame([feature_values], columns=feature_names)
    predicted_score = float(model.predict(feature_df)[0])

    # Rough confidence bands based on UCI UPDRS range (0-55)
    if 0 <= predicted_score <= 15:
        confidence_band = "high (low UPDRS)"
    elif 15 < predicted_score <= 30:
        confidence_band = "medium"
    else:
        confidence_band = "high (high UPDRS)"

    return UPDRSPrediction(
        call_id=features.call_id,
        predicted_score=round(predicted_score, 1),
        confidence_band=confidence_band,
        anomaly_flag=False,
    )
