"""
Fixed call state machine — Zone A.

Defines the sequence of tasks in a check-in call. agent.py drives through
these states in order; each state owns the prompt(s) it needs to say and
knows what kind of answer it's expecting (so triggers.py knows how to check
it and ml/audio.py knows what kind of audio segment to extract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class CallState(Enum):
    RECALL_CHECK = auto()          # "What did you have for breakfast?" etc — checked against baseline
    SUSTAINED_PHONATION = auto()   # "Say 'ahhh' for as long as you can" — jitter/shimmer/HNR source
    READING_TASK = auto()          # read a fixed passage — speech rate / pause features
    OPEN_QA = auto()               # free-form check-in questions — symptom-flag trigger surface
    COUNTING_TASK = auto()         # "count from 1 to 20" — pause/rhythm features
    COMPLETE = auto()


# TODO(agent): fill in real prompt copy / rotate question banks per call.
STATE_PROMPTS: dict[CallState, list[str]] = {
    CallState.RECALL_CHECK: ["What did you have for breakfast this morning?"],
    CallState.SUSTAINED_PHONATION: ["Take a deep breath and say 'ahh' for as long as you comfortably can."],
    CallState.READING_TASK: ["Please read this sentence aloud: TODO(agent) insert fixed reading passage."],
    CallState.OPEN_QA: ["How have you been feeling since we last spoke?"],
    CallState.COUNTING_TASK: ["Please count out loud from one to twenty."],
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


@dataclass
class CallScript:
    state: CallState = CallState.RECALL_CHECK
    prompt_index: int = 0
    # question_id counter, used to key MossQARecord.question_id
    turn_count: int = 0

    def current_prompts(self) -> list[str]:
        return STATE_PROMPTS[self.state]

    def current_question_id(self) -> str:
        return f"{self.state.name.lower()}:{self.prompt_index}"

    def is_complete(self) -> bool:
        return self.state == CallState.COMPLETE

    def advance(self) -> CallState:
        """Move to the next state in the fixed sequence."""
        self.turn_count += 1
        idx = _ORDER.index(self.state)
        self.state = _ORDER[min(idx + 1, len(_ORDER) - 1)]
        self.prompt_index = 0
        return self.state
