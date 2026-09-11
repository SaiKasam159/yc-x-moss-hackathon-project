"""
Shared data contracts between Zone A (voice agent) and Zone B (ML pipeline).

These dataclasses mirror db/schema.sql exactly. Do not rename fields or change
types without confirming with both workstream owners — both /agent and /ml
import from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Patient:
    name: str
    id: Optional[int] = None
    baseline_call_id: Optional[int] = None
    created_at: Optional[str] = None  # ISO 8601, set by DB default if omitted


@dataclass
class Call:
    patient_id: int
    id: Optional[int] = None
    timestamp: Optional[str] = None  # ISO 8601, set by DB default if omitted
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None


@dataclass
class FeatureVector:
    """Acoustic + prosodic features for one call, matching the UCI Parkinson's
    Telemonitoring dataset columns used to train the UPDRS regression model."""

    call_id: int
    jitter_local: float
    jitter_rap: float
    shimmer_local: float
    shimmer_apq5: float
    hnr: float
    rpde: float
    dfa: float
    ppe: float
    speech_rate: float
    pause_freq: float
    pause_avg_duration: float


@dataclass
class UPDRSPrediction:
    call_id: int
    predicted_score: float
    confidence_band: Optional[str] = None
    anomaly_flag: bool = False


@dataclass
class MossQARecord:
    """One question/answer turn from a call, scoped for retrieval in Moss.
    Ingested per-turn during a call, and queried scoped to
    (patient_id[, question_topic]) for real-time mid-call retrieval."""

    patient_id: int
    call_id: int
    question_id: str
    question_topic: str
    answer_text: str
    timestamp: str  # ISO 8601

    # Free-form metadata bag for anything not worth a dedicated column yet
    # (e.g. detected sentiment, confidence score from STT). Not persisted to
    # SQLite — Moss-only.
    extra: dict = field(default_factory=dict)
