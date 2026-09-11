"""
UPDRS prediction — Zone B.

STUB: explicitly returns a fake prediction until train_model.py has produced
a real ml/model.pkl. /agent and /reports are built against this contract
(db.contracts.UPDRSPrediction) now, so swapping in the real model later is a
drop-in change to predict_updrs() only — do not change the return type or
agent.py's expectations to work around this stub.
"""

from __future__ import annotations

import random

from db.contracts import FeatureVector, UPDRSPrediction

MODEL_PATH = "ml/model.pkl"


def predict_updrs(features: FeatureVector) -> UPDRSPrediction:
    """STUB — returns a fake but plausible UPDRS prediction, ignoring the
    input features entirely. TODO(ml): once ml/model.pkl exists (see
    train_model.py), load it and predict for real; keep the return type
    exactly UPDRSPrediction.
    """
    fake_score = round(random.uniform(10.0, 40.0), 1)  # UCI total_UPDRS range is roughly 0-55
    return UPDRSPrediction(
        call_id=features.call_id,
        predicted_score=fake_score,
        confidence_band="low (stub model)",
        anomaly_flag=False,
    )
