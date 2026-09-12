"""
Mid-call trigger checks — Zone A.

Two kinds of triggers decide whether the agent should break from the fixed
script and run a Moss retrieval + follow-up question:

1. Deterministic wrong-answer check — compares a RECALL_CHECK answer
   (orientation questions like day-of-week/season, or a personal-recall
   question) against a known correct/expected value.
2. Semantic symptom-flag check — scans free-form OPEN_QA answers for
   language suggesting a flagged symptom, worth a clarifying follow-up.
   Implemented as a lightweight keyword match against a small Parkinson's
   symptom vocabulary (word-boundary matching, not substring), with an
   embedding-similarity upgrade path noted below for when that's worth the
   extra latency/dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Category -> surface forms. Keep this small and hand-curated for the demo;
# swap/extend once real call transcripts show what patients actually say.
SYMPTOM_VOCABULARY: dict[str, list[str]] = {
    "tremor": ["tremor", "tremors", "shaking", "shake", "shakes", "trembling", "tremble"],
    "stiffness": ["stiff", "stiffness", "rigid", "rigidity"],
    "balance": ["balance", "unsteady", "fell", "falling", "falls", "dizzy", "dizziness"],
    "freezing": ["freezing", "froze", "frozen", "stuck", "locked up"],
    "fatigue": ["tired", "fatigue", "fatigued", "exhausted", "no energy", "low energy", "worn out"],
    "medication issues": [
        "medication", "medications", "meds", "pills", "dose", "dosage",
        "side effect", "side effects", "forgot to take", "missed a dose", "ran out of",
    ],
}

_WORD_RE_CACHE: dict[str, re.Pattern] = {}


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Word-boundary regex for a keyword/short phrase (avoids matching
    'stiff' inside an unrelated word, unlike plain substring search)."""
    if phrase not in _WORD_RE_CACHE:
        _WORD_RE_CACHE[phrase] = re.compile(r"\b" + re.escape(phrase) + r"\b")
    return _WORD_RE_CACHE[phrase]


@dataclass
class TriggerResult:
    fired: bool
    reason: Optional[str] = None
    query_text: Optional[str] = None  # text to hand to moss_client.query_patient_history


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


# Accepted alternatives for answers that have more than one correct wording.
# Without these a patient saying "autumn" when we expect "fall" is marked
# wrong and gets an unnecessary follow-up — a false alarm that wastes the
# patient's time and pollutes the flags a clinician sees.
_ANSWER_SYNONYMS: dict[str, set[str]] = {
    "fall": {"fall", "autumn"},
    "autumn": {"fall", "autumn"},
}

# Negations: "I had toast, NOT eggs" contains "eggs", so a plain substring
# check would score it correct. If the expected answer only appears negated,
# treat it as a miss.
_NEGATORS = ("not", "no", "never", "didnt", "didn t", "dont", "don t", "wasnt", "isnt")


def _mentions_expected(answer: str, expected: str) -> bool:
    """Is `expected` (or an accepted synonym) genuinely present, not negated?"""
    candidates = _ANSWER_SYNONYMS.get(expected, {expected})
    for cand in candidates:
        match = _phrase_pattern(cand).search(answer)
        if not match:
            continue
        # Look at the few words immediately before the match for a negator.
        preceding = answer[: match.start()].split()[-3:]
        if any(tok in _NEGATORS for tok in preceding):
            continue  # present, but negated — keep looking
        return True
    return False


def check_wrong_answer(answer_text: str, expected_answer: Optional[str]) -> TriggerResult:
    """Deterministic check: does this answer contradict a known expected
    value? Used for RECALL_CHECK prompts — orientation questions (expected
    answer computed dynamically, see call_script.orientation_expected_answer)
    and the personal-recall question (expected answer from the patient's
    baseline call).

    Accepts synonyms and rejects negated mentions, so ordinary phrasing
    ("autumn" for "fall", "not eggs" for "eggs") isn't mis-scored.
    """
    if expected_answer is None:
        return TriggerResult(fired=False)

    if not _mentions_expected(_normalize(answer_text), _normalize(expected_answer)):
        return TriggerResult(
            fired=True,
            reason=f"answer did not match expected value: {expected_answer!r}",
            query_text=answer_text,
        )
    return TriggerResult(fired=False)


def check_symptom_flag(answer_text: str) -> TriggerResult:
    """Semantic check: does this free-form answer mention a flagged symptom?

    Lightweight keyword match (word-boundary, case-insensitive) against
    SYMPTOM_VOCABULARY. TODO(agent): upgrade path is an embedding-similarity
    check against a small reference set of symptom sentences instead of
    exact keywords — worth it once false negatives on paraphrased symptoms
    (e.g. "my hands won't stop moving" for tremor) start to matter; keyword
    matching is intentionally the cheap/fast first pass so this never adds
    retrieval-blocking latency mid-call.
    """
    hits: list[str] = []
    for category, phrases in SYMPTOM_VOCABULARY.items():
        for phrase in phrases:
            if _phrase_pattern(phrase).search(answer_text.lower()):
                hits.append(f"{category}:{phrase}")
                break  # one hit per category is enough to flag it

    if hits:
        return TriggerResult(
            fired=True,
            reason=f"symptom keywords matched: {hits}",
            query_text=answer_text,
        )
    return TriggerResult(fired=False)
