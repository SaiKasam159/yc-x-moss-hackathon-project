"""
Mid-call trigger checks — Zone A.

Two kinds of triggers decide whether the agent should break from the fixed
script and run a Moss retrieval + follow-up question:

1. Deterministic wrong-answer check — compares an answer against a known
   baseline/expected value (e.g. recall-check answers vs. the patient's
   baseline call).
2. Semantic symptom-flag check — scans free-form answers (OPEN_QA) for
   language suggesting a flagged symptom, worth a clarifying follow-up.

Both are stubs: deterministic logic and a keyword list respectively, to be
replaced/tuned once real call data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# TODO(agent): replace with a real symptom lexicon / classifier.
_SYMPTOM_KEYWORDS = [
    "dizzy", "dizziness", "fell", "falling", "tremor", "shaking",
    "stiff", "stiffness", "confused", "can't sleep", "trouble sleeping",
    "pain", "numb", "numbness",
]


@dataclass
class TriggerResult:
    fired: bool
    reason: Optional[str] = None
    query_text: Optional[str] = None  # text to hand to moss_client.query_patient_history


def check_wrong_answer(answer_text: str, expected_answer: Optional[str]) -> TriggerResult:
    """Deterministic check: does this answer contradict a known expected value?

    `expected_answer` is typically pulled from the patient's baseline call or
    a fixed fact (e.g. today's date, a previously stated detail). Stubbed as
    an exact/substring mismatch for now.
    """
    if expected_answer is None:
        return TriggerResult(fired=False)

    if expected_answer.strip().lower() not in answer_text.strip().lower():
        return TriggerResult(
            fired=True,
            reason=f"answer did not match expected value: {expected_answer!r}",
            query_text=answer_text,
        )
    return TriggerResult(fired=False)


def check_symptom_flag(answer_text: str) -> TriggerResult:
    """Semantic check: does this free-form answer mention a flagged symptom?

    Stubbed as keyword matching. TODO(agent): swap for an LLM classifier or
    embedding-similarity check against a symptom reference set.
    """
    lowered = answer_text.lower()
    hits = [kw for kw in _SYMPTOM_KEYWORDS if kw in lowered]
    if hits:
        return TriggerResult(
            fired=True,
            reason=f"symptom keywords matched: {hits}",
            query_text=answer_text,
        )
    return TriggerResult(fired=False)
