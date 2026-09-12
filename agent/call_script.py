"""
Fixed call state machine — Zone A.

Defines the sequence of tasks in a check-in call. agent.py drives through
these states in order; each state owns the prompt(s) it needs to say and
knows what kind of answer it's expecting (so triggers.py knows how to check
it and ml/audio.py knows what kind of audio segment to extract).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class CallState(Enum):
    RECALL_CHECK = auto()          # orientation + personal recall — checked against known/baseline answers
    SUSTAINED_PHONATION = auto()   # "Say 'ahhh' for as long as you can" — jitter/shimmer/HNR source
    READING_TASK = auto()          # read a fixed passage — speech rate / pause features
    OPEN_QA = auto()               # free-form check-in questions — symptom-flag trigger surface
    COUNTING_TASK = auto()         # "count from 1 to 20" — pause/rhythm features
    COMPLETE = auto()


# A short, original passage (not a copyrighted clinical instrument like the
# Rainbow Passage) — swap for a licensed one later if desired.
READING_PASSAGE = (
    "The old dog stretched slowly in the warm afternoon sun. "
    "A light breeze moved through the open window, carrying the smell of "
    "fresh coffee from the kitchen down the hall."
)

STATE_PROMPTS: dict[CallState, list[str]] = {
    CallState.RECALL_CHECK: [
        "What day of the week is it today?",           # orientation — dynamic expected answer
        "What season is it right now?",                # orientation — dynamic expected answer
        "What did you have for breakfast this morning?",  # personal recall — baseline-derived expected answer
    ],
    CallState.SUSTAINED_PHONATION: [
        "Take a deep breath and say 'ahh' for as long as you comfortably can.",
    ],
    CallState.READING_TASK: [
        f"Please read this sentence aloud: {READING_PASSAGE}",
    ],
    CallState.OPEN_QA: [
        "How have you been feeling since we last spoke?",
        "Have you noticed any changes in your daily activities lately?",
    ],
    CallState.COUNTING_TASK: [
        "Please count out loud from one to twenty.",
    ],
    CallState.COMPLETE: [],
}

_ORDER = [
    CallState.RECALL_CHECK,
    CallState.SUSTAINED_PHONATION,
    CallState.READING_TASK,
    CallState.OPEN_QA,
    CallState.COUNTING_TASK,
    CallState.COMPLETE,
]

_SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}


def orientation_expected_answer(question_id: str, today: Optional[datetime.date] = None) -> Optional[str]:
    """Dynamically computed expected answers for the orientation-type prompts
    in RECALL_CHECK (today's day-of-week / season). Returns None for
    question_ids that aren't orientation questions — e.g. the personal
    recall prompt, whose expected answer comes from the patient's baseline
    instead (see agent.py's TurnContext.expected_answers).
    """
    today = today or datetime.date.today()
    if question_id == "recall_check:0":
        return today.strftime("%A")
    if question_id == "recall_check:1":
        return _SEASON_BY_MONTH[today.month]
    return None


@dataclass
class CallScript:
    state: CallState = CallState.RECALL_CHECK
    prompt_index: int = 0
    turn_count: int = 0

    def current_prompt(self) -> Optional[str]:
        """The prompt to say right now, or None if the call is complete."""
        prompts = STATE_PROMPTS[self.state]
        if self.prompt_index >= len(prompts):
            return None
        return prompts[self.prompt_index]

    def current_question_id(self) -> str:
        return f"{self.state.name.lower()}:{self.prompt_index}"

    def is_complete(self) -> bool:
        return self.state == CallState.COMPLETE

    def advance(self) -> Optional[str]:
        """Move to the next prompt — either the next prompt within the
        current state, or the first prompt of the next state if the current
        state is exhausted. Returns the new current prompt (None if the call
        just completed)."""
        self.turn_count += 1
        prompts = STATE_PROMPTS[self.state]
        if self.prompt_index + 1 < len(prompts):
            self.prompt_index += 1
        else:
            idx = _ORDER.index(self.state)
            self.state = _ORDER[min(idx + 1, len(_ORDER) - 1)]
            self.prompt_index = 0
        return self.current_prompt()
