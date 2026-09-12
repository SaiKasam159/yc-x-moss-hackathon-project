"""
Feature vector assembly — Zone B.

Combines outputs of ml/audio.py into a db.contracts.FeatureVector matching
the UCI Parkinson's Telemonitoring dataset columns. rpde/dfa/ppe are the
nonlinear dynamics measures (not covered by parselmouth); they're now
implemented in ml/nonlinear.py and extracted by ml/audio.py — see that
module's calibration caveat about comparing them to the UCI columns.
"""

from __future__ import annotations

from db.contracts import FeatureVector


def build_feature_vector(call_id: int, acoustic: dict, prosody: dict) -> FeatureVector:
    """Assemble a FeatureVector from the dicts returned by
    ml.audio.extract_acoustic_features / extract_prosody_features.

    Missing keys fall back to 0.0 so a partial extraction still yields a
    usable vector; ml/audio.py logs a warning whenever it has to substitute
    a fallback, so a silently-constant feature is visible rather than
    quietly feeding the model.
    """
    return FeatureVector(
        call_id=call_id,
        jitter_local=acoustic.get("jitter_local", 0.0),
        jitter_rap=acoustic.get("jitter_rap", 0.0),
        shimmer_local=acoustic.get("shimmer_local", 0.0),
        shimmer_apq5=acoustic.get("shimmer_apq5", 0.0),
        hnr=acoustic.get("hnr", 0.0),
        rpde=acoustic.get("rpde", 0.0),
        dfa=acoustic.get("dfa", 0.0),
        ppe=acoustic.get("ppe", 0.0),
        speech_rate=prosody.get("speech_rate", 0.0),
        pause_freq=prosody.get("pause_freq", 0.0),
        pause_avg_duration=prosody.get("pause_avg_duration", 0.0),
    )
