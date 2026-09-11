"""
Feature vector assembly — Zone B.

Combines outputs of ml/audio.py into a db.contracts.FeatureVector matching
the UCI Parkinson's Telemonitoring dataset columns (plus rpde/dfa/ppe, which
are nonlinear dynamics measures also present in that dataset and required by
the trained model — TODO(ml): implement their extraction, likely via nolds
or a similar library; not covered by parselmouth).
"""

from __future__ import annotations

from db.contracts import FeatureVector


def build_feature_vector(call_id: int, acoustic: dict, prosody: dict) -> FeatureVector:
    """Assemble a FeatureVector from the dicts returned by
    ml.audio.extract_acoustic_features / extract_prosody_features.

    TODO(ml): once rpde/dfa/ppe extraction exists, merge that in too —
    currently defaulted to 0.0 placeholders so this is at least callable
    end-to-end against the stub audio functions.
    """
    return FeatureVector(
        call_id=call_id,
        jitter_local=acoustic.get("jitter_local", 0.0),
        jitter_rap=acoustic.get("jitter_rap", 0.0),
        shimmer_local=acoustic.get("shimmer_local", 0.0),
        shimmer_apq5=acoustic.get("shimmer_apq5", 0.0),
        hnr=acoustic.get("hnr", 0.0),
        rpde=acoustic.get("rpde", 0.0),   # TODO(ml): nonlinear dynamics — not yet extracted
        dfa=acoustic.get("dfa", 0.0),     # TODO(ml): nonlinear dynamics — not yet extracted
        ppe=acoustic.get("ppe", 0.0),     # TODO(ml): nonlinear dynamics — not yet extracted
        speech_rate=prosody.get("speech_rate", 0.0),
        pause_freq=prosody.get("pause_freq", 0.0),
        pause_avg_duration=prosody.get("pause_avg_duration", 0.0),
    )
