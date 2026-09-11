"""
Moss client — ingestion + scoped semantic query for per-patient, per-question
retrieval during a live call.

Real implementation uses the `moss` Python SDK (https://docs.moss.dev),
client-level `create_index`/`add_docs`/`load_index`/`query` — one cloud
index per patient, named f"patient:{id}", persisted immediately on every
write (no separate "push" step). See "why not SessionIndex" below.

Needs MOSS_PROJECT_ID / MOSS_PROJECT_KEY (see .env.example) — get these from
https://portal.usemoss.dev. Until both are set, this module runs in STUB
MODE: an in-memory list stands in for the Moss index, with naive keyword
overlap standing in for semantic similarity. This keeps /agent runnable
end-to-end without live credentials.

NOTE: ingest_qa_record() and query_patient_history() are `async def` here —
the real moss SDK is async-only. This is a signature change from the earlier
stub (which was sync); agent.py has been updated accordingly.

Why not SessionIndex (moss's own real-time-session API, which looks like a
better fit on paper): tested it directly against the real project and found
a reproducible bug in this SDK version (1.9.1) — a session seeded via
create_index() then mutated with several add_docs() calls and finally
push_index() ends up with the cloud index's model silently degraded to
"custom" (which then requires caller-supplied embeddings and breaks every
future query, since none are ever supplied). client-level create_index() /
add_docs() / load_index() / query() does not hit this bug — verified with 8
sequential writes followed by a real semantic query. Tradeoff: each
ingest_qa_record() write is now a real network call rather than local/
in-memory, so mid-call latency is a little higher than SessionIndex would
have given (had it worked) — acceptable for this project's timeout-based
fallback design (see agent.py's FALLBACK_TIMEOUT_S), but worth revisiting if
Moss ships a fix and SessionIndex becomes viable again.
"""

from __future__ import annotations

import os
from typing import Optional

from db.contracts import MossQARecord

MOSS_PROJECT_ID = os.environ.get("MOSS_PROJECT_ID")
MOSS_PROJECT_KEY = os.environ.get("MOSS_PROJECT_KEY")

# True until real credentials are present, so `agent.py` and friends can be
# exercised locally before Moss keys exist.
STUB_MODE = not (MOSS_PROJECT_ID and MOSS_PROJECT_KEY)

# In-memory stand-in for the Moss index while STUB_MODE is on.
_stub_store: list[MossQARecord] = []

# Real mode: one MossClient; track which patient indexes exist / are loaded
# so we don't redundantly create or load_index() on every call.
_client = None
_known_indexes: set[int] = set()   # patient_ids confirmed to have a cloud index
_loaded_indexes: set[int] = set()  # patient_ids currently load_index()'d


def _index_name(patient_id: int) -> str:
    return f"patient:{patient_id}"


def _get_client():
    global _client
    if _client is None:
        from moss import MossClient
        _client = MossClient(MOSS_PROJECT_ID, MOSS_PROJECT_KEY)
    return _client


async def _index_exists(client, name: str) -> bool:
    try:
        await client.get_index(name)
        return True
    except Exception:
        return False


async def _to_document(record: MossQARecord):
    from moss import DocumentInfo
    return DocumentInfo(
        id=f"{record.call_id}:{record.question_id}",
        text=record.answer_text,
        metadata={
            "patient_id": str(record.patient_id),
            "call_id": str(record.call_id),
            "question_id": record.question_id,
            "question_topic": record.question_topic,
            "timestamp": record.timestamp,
        },
    )


async def ingest_qa_record(record: MossQARecord) -> None:
    """Embed + upsert one QA turn into Moss, scoped to the patient.

    Called by /agent right after each answer is captured, so it's retrievable
    for the rest of the call (and future calls — every write here is already
    persisted to the cloud, no separate push step needed).
    """
    if STUB_MODE:
        _stub_store.append(record)
        return

    from moss import MutationOptions

    client = _get_client()
    name = _index_name(record.patient_id)
    doc = await _to_document(record)

    if record.patient_id in _known_indexes or await _index_exists(client, name):
        _known_indexes.add(record.patient_id)
        await client.add_docs(name, [doc], MutationOptions(upsert=True))
    else:
        # Brand-new patient — create_index() (not session()) so the model is
        # set correctly; see module docstring for why session() is avoided.
        await client.create_index(name, [doc], model_id="moss-minilm")
        _known_indexes.add(record.patient_id)

    _loaded_indexes.discard(record.patient_id)  # stale after a write; reload before next query


async def query_patient_history(
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

    from moss import QueryOptions

    client = _get_client()
    name = _index_name(patient_id)

    if patient_id not in _known_indexes:
        if not await _index_exists(client, name):
            return []  # no history yet for this patient
        _known_indexes.add(patient_id)

    if patient_id not in _loaded_indexes:
        await client.load_index(name)
        _loaded_indexes.add(patient_id)

    query_filter = (
        {"field": "question_topic", "condition": {"$eq": question_topic}}
        if question_topic else None
    )
    result = await client.query(name, query_text, QueryOptions(top_k=top_k, filter=query_filter))
    return [_record_from_moss_doc(patient_id, doc) for doc in result.docs]


def _record_from_moss_doc(patient_id: int, doc) -> MossQARecord:
    meta = doc.metadata or {}
    return MossQARecord(
        patient_id=patient_id,
        call_id=int(meta.get("call_id", 0)),
        question_id=meta.get("question_id", doc.id),
        question_topic=meta.get("question_topic", ""),
        answer_text=doc.text,
        timestamp=meta.get("timestamp", ""),
        extra={"score": doc.score},
    )


async def push_patient_session(patient_id: int) -> None:
    """No-op in the current (non-SessionIndex) implementation — every
    ingest_qa_record() write is already persisted to the cloud immediately.
    Kept as a function (rather than removed) so agent.py's existing
    end-of-call call site doesn't need touching, and so re-introducing
    SessionIndex later (if Moss fixes the bug noted in the module docstring)
    is a swap here, not a call-site change.
    """
    return


def reset_stub_store() -> None:
    """Test helper: clear the in-memory stub store between test runs."""
    _stub_store.clear()
