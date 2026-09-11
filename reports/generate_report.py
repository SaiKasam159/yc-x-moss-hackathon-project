"""
Structured report generation — shared surface (reads from both zones' output).

Turns a FeatureVector + call transcript + UPDRS prediction + trigger flags
into a structured human-readable report, and pushes a summary record to Moss
so future calls can retrieve "last report said X" context.

STUB: no real LLM call wired yet — provider TBD (see README "Open questions").
"""

from __future__ import annotations

from dataclasses import dataclass

from db.contracts import FeatureVector, MossQARecord, UPDRSPrediction
from db.moss_client import ingest_qa_record


@dataclass
class CallReport:
    call_id: int
    patient_id: int
    summary_text: str
    predicted_score: float
    anomaly_flag: bool


def generate_report(
    patient_id: int,
    features: FeatureVector,
    transcript: str,
    prediction: UPDRSPrediction,
    flags: list[str],
) -> CallReport:
    """TODO(reports): real LLM call to synthesize a structured report from
    the transcript + extracted features + prediction + any trigger flags
    raised during the call. Stubbed with a templated summary for now.
    """
    summary_text = (
        f"[STUB REPORT] Call {features.call_id} for patient {patient_id}: "
        f"predicted UPDRS {prediction.predicted_score} ({prediction.confidence_band}). "
        f"Flags raised: {flags or 'none'}. "
        f"TODO(reports): replace with real LLM-generated summary from transcript."
    )
    return CallReport(
        call_id=features.call_id,
        patient_id=patient_id,
        summary_text=summary_text,
        predicted_score=prediction.predicted_score,
        anomaly_flag=prediction.anomaly_flag,
    )


def push_report_to_moss(report: CallReport) -> None:
    """Push the report summary into Moss as a retrievable record, so a future
    call's mid-call retrieval can surface "your last report noted X"."""
    record = MossQARecord(
        patient_id=report.patient_id,
        call_id=report.call_id,
        question_id="report_summary",
        question_topic="report",
        answer_text=report.summary_text,
        timestamp=__import__("datetime").datetime.utcnow().isoformat() + "Z",
    )
    ingest_qa_record(record)
