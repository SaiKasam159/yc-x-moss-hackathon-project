"""
Moss client — ingestion + scoped semantic query for per-patient, per-question
retrieval during a live call.

Real Moss wiring is TODO (needs MOSS_API_KEY / MOSS_ENDPOINT — see .env.example).
Until those are set, this module runs in STUB MODE: an in-memory list stands
in for the Moss index, with naive keyword overlap standing in for semantic
similarity. This keeps /agent runnable end-to-end without live credentials.

Swap points for the real integration are marked with TODO(moss).
"""

from __future__ import annotations

import os
from typing import Optional

from db.contracts import MossQARecord

MOSS_API_KEY = os.environ.get("MOSS_API_KEY")
MOSS_ENDPOINT = os.environ.get("MOSS_ENDPOINT")

# True until real credentials are present, so `agent.py` and friends can be
# exercised locally before Moss keys exist.
STUB_MODE = not (MOSS_API_KEY and MOSS_ENDPOINT)

# In-memory stand-in for the Moss index while STUB_MODE is on.
_stub_store: list[MossQARecord] = []


def ingest_qa_record(record: MossQARecord) -> None:
    """Embed + upsert one QA turn into Moss, scoped to the patient.

    Called by /agent right after each answer is captured, so it's retrievable
    for the rest of the call (and future calls).
    """
    if STUB_MODE:
        _stub_store.append(record)
        return

    # TODO(moss): real ingestion call, e.g.
    #   moss.upsert(
    #       namespace=f"patient:{record.patient_id}",
    #       id=f"{record.call_id}:{record.question_id}",
    #       text=record.answer_text,
    #       metadata={
    #           "patient_id": record.patient_id,
    #           "call_id": record.call_id,
    #           "question_id": record.question_id,
    #           "question_topic": record.question_topic,
    #           "timestamp": record.timestamp,
    #       },
    #   )
    raise NotImplementedError("Real Moss ingestion not wired yet — set MOSS_API_KEY/MOSS_ENDPOINT and implement.")


def query_patient_history(
    patient_id: int,
    query_text: str,
    question_topic: Optional[str] = None,
    top_k: int = 5,
) -> list[MossQARecord]:
    """Scoped semantic retrieval: most relevant past answers for this patient,
    optionally narrowed to a question_topic. This is the call made mid-call
    (between STT and LLM response) when a trigger fires.
    """
    if STUB_MODE:
        candidates = [
            r for r in _stub_store
            if r.patient_id == patient_id
            and (question_topic is None or r.question_topic == question_topic)
        ]
        # Naive keyword-overlap "similarity" ranking as a placeholder for
        # real semantic search.
        query_terms = set(query_text.lower().split())

        def _score(rec: MossQARecord) -> int:
            return len(query_terms & set(rec.answer_text.lower().split()))

        candidates.sort(key=_score, reverse=True)
        return candidates[:top_k]

    # TODO(moss): real scoped query call, e.g.
    #   results = moss.query(
    #       namespace=f"patient:{patient_id}",
    #       text=query_text,
    #       filter={"question_topic": question_topic} if question_topic else None,
    #       top_k=top_k,
    #   )
    #   return [_record_from_moss_result(r) for r in results]
    raise NotImplementedError("Real Moss query not wired yet — set MOSS_API_KEY/MOSS_ENDPOINT and implement.")


def reset_stub_store() -> None:
    """Test helper: clear the in-memory stub store between test runs."""
    _stub_store.clear()
