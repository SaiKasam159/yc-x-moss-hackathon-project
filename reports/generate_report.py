"""
Deterministic report generation — shared surface (reads from both zones' output).

Turns a FeatureVector + call transcript + UPDRS prediction + trigger flags
into a structured human-readable report using templates, and pushes a summary
record to Moss so future calls can retrieve "last report said X" context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from db.contracts import FeatureVector, MossQARecord, UPDRSPrediction
from db.moss_client import ingest_qa_record


@dataclass
class CallReport:
    call_id: int
    patient_id: int
    summary_text: str
    predicted_score: float
    anomaly_flag: bool


ALERT_TEMPLATES = {
    "elevated_jitter": "Elevated jitter detected — voice may indicate tremor or vocal instability.",
    "elevated_shimmer": "Elevated shimmer detected — voice quality variation noted.",
    "reduced_hnr": "Reduced harmonic-to-noise ratio — voice quality degradation.",
    "elevated_pause_freq": "Frequent pauses during speech — may indicate cognitive or motor changes.",
    "low_speech_rate": "Slowed speech rate — possible dysarthria or cognitive slowness.",
    "high_updrs_pred": "UPDRS score trending higher — monitor symptom progression.",
    "low_updrs_pred": "UPDRS score improved — positive trajectory.",
}


def generate_report(
    patient_id: int,
    features: FeatureVector,
    transcript: str,
    prediction: UPDRSPrediction,
    flags: list[str],
) -> CallReport:
    """Generate a deterministic report from extracted features and prediction.

    Args:
        patient_id: Patient ID
        features: Extracted acoustic/prosody features
        transcript: Call transcript (for context/logging, not for analysis)
        prediction: UPDRS prediction from model
        flags: List of detected anomaly flags

    Returns:
        CallReport with templated summary text
    """
    # Measured first, modelled last. What follows above the score is measured
    # directly from the patient's voice; the score is a model output that is
    # not validated for a patient it hasn't heard (see ml/predict.py), so it
    # must not be the headline.
    sections = [f"Patient {patient_id} | Call {features.call_id}"]

    if flags:
        alert_texts = [ALERT_TEMPLATES.get(f, f"Flag: {f}") for f in flags]
        sections.append("Raised during the call:\n  • " + "\n  • ".join(alert_texts))

    # Units matter here: Praat's local jitter and shimmer are fractions, so
    # they're shown as percentages. They were previously labelled "%" without
    # converting, and shimmer was labelled "dB", which it isn't.
    sections.append(
        "Voice (measured): "
        f"jitter {features.jitter_local * 100:.2f}% (RAP {features.jitter_rap * 100:.2f}%), "
        f"shimmer {features.shimmer_local * 100:.2f}% (APQ5 {features.shimmer_apq5 * 100:.2f}%), "
        f"HNR {features.hnr:.1f} dB"
    )
    sections.append(
        "Nonlinear (measured): "
        f"RPDE {features.rpde:.3f}, DFA {features.dfa:.3f}, PPE {features.ppe:.3f}"
    )
    sections.append(
        "Speech (measured): "
        f"{features.speech_rate:.0f} words/min, "
        f"{features.pause_freq:.2f} pauses/s, "
        f"average pause {features.pause_avg_duration:.2f}s"
    )
    sections.append(
        f"Model estimate (experimental): UPDRS {prediction.predicted_score} "
        f"— {prediction.confidence_band}."
    )
    if prediction.anomaly_flag:
        sections.append(
            "  Note: this call's measurements fall outside the range the model "
            "was trained on, so the estimate is extrapolation."
        )
    sections.append(
        "Compare the measured values with this patient's own previous calls; "
        "the model estimate is not a clinical measurement."
    )

    summary_text = "\n".join(sections)

    return CallReport(
        call_id=features.call_id,
        patient_id=patient_id,
        summary_text=summary_text,
        predicted_score=prediction.predicted_score,
        anomaly_flag=prediction.anomaly_flag,
    )


async def push_report_to_moss(report: CallReport) -> None:
    """Push the report summary into Moss as a retrievable record, so a future
    call's mid-call retrieval can surface "your last report noted X".

    Must be awaited. This used to call the async ingest_qa_record without
    awaiting it, so the coroutine was created, dropped, and nothing was ever
    written.
    """
    record = MossQARecord(
        patient_id=report.patient_id,
        call_id=report.call_id,
        question_id="report_summary",
        question_topic="report",
        answer_text=report.summary_text,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
    await ingest_qa_record(record)
