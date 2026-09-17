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
from zoneinfo import ZoneInfo


class CallState(Enum):
    RECALL_CHECK = auto()          # orientation + personal recall — checked against known/baseline answers
    SUSTAINED_PHONATION = auto()   # "Say 'ahhh' for as long as you can" — jitter/shimmer/HNR source
    READING_TASK = auto()          # read a fixed passage — speech rate / pause features
    OPEN_QA = auto()               # free-form check-in questions — symptom-flag trigger surface
    COUNTING_TASK = auto()         # "count from 1 to 20" — pause/rhythm features
    DELAYED_RECALL = auto()        # the three words from the start of the call
    COMPLETE = auto()


class CaptureMode(Enum):
    """How the agent listens during a stage."""
    ANSWER = auto()       # one spoken answer per prompt; triggers may follow up
    TIMED_AUDIO = auto()  # not speech: record the microphone, don't wait for a transcript
    LONG_SPEECH = auto()  # extended speech: keep collecting until the patient stops


CAPTURE_MODES: dict[CallState, CaptureMode] = {
    CallState.RECALL_CHECK: CaptureMode.ANSWER,
    CallState.SUSTAINED_PHONATION: CaptureMode.TIMED_AUDIO,
    CallState.READING_TASK: CaptureMode.LONG_SPEECH,
    CallState.OPEN_QA: CaptureMode.ANSWER,
    CallState.COUNTING_TASK: CaptureMode.LONG_SPEECH,
    CallState.DELAYED_RECALL: CaptureMode.ANSWER,
}

# Stretches of patient.wav cut out for the ML pipeline: the phonation segment
# feeds jitter/shimmer/HNR/RPDE/DFA/PPE; reading and counting feed speech rate
# and pause features.
SEGMENT_NAMES: dict[CallState, str] = {
    CallState.SUSTAINED_PHONATION: "phonation",
    CallState.READING_TASK: "reading",
    CallState.COUNTING_TASK: "counting",
}


# Three unrelated everyday words, said early in the call and asked for again
# at the end. This replaced "what did you have for breakfast?", which looked
# like a memory check but wasn't: breakfast changes daily, so no baseline
# answer can be right, and anyone who ate something different was flagged.
# These the agent chooses, so it knows the correct answer.
RECALL_WORD_LISTS: tuple[tuple[str, str, str], ...] = (
    ("apple", "table", "penny"),
    ("carrot", "window", "river"),
    ("book", "garden", "silver"),
    ("lemon", "chair", "cloud"),
)


def recall_words_for_call(call_id: int) -> tuple[str, str, str]:
    """Rotate the list per call so a patient doesn't simply learn one set."""
    return RECALL_WORD_LISTS[call_id % len(RECALL_WORD_LISTS)]


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
        # Registration for the delayed recall at the end of the call.
        "I'm going to say three words, and I'll ask you for them again later: "
        "{words}. Can you repeat them back to me now?",
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
    CallState.DELAYED_RECALL: [
        "Earlier I asked you to remember three words. Can you tell me what they were?",
    ],
    CallState.COMPLETE: [],
}

_ORDER = [
    CallState.RECALL_CHECK,
    CallState.SUSTAINED_PHONATION,
    CallState.READING_TASK,
    CallState.OPEN_QA,
    CallState.COUNTING_TASK,
    CallState.DELAYED_RECALL,
    CallState.COMPLETE,
]

_SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

# Astronomical season starts (approximate; the solstice/equinox moves a day
# either way between years).
_ASTRONOMICAL_STARTS = [(3, 20, "spring"), (6, 21, "summer"), (9, 22, "fall"), (12, 21, "winter")]

_OPPOSITE_SEASON = {"winter": "summer", "summer": "winter", "spring": "fall", "fall": "spring"}

_ZONE_TAB_PATHS = ("/usr/share/zoneinfo/zone1970.tab", "/usr/share/zoneinfo/zone.tab")
# Used only if the tz database isn't readable.
_SOUTHERN_PREFIXES = (
    "Australia/", "Antarctica/", "Pacific/Auckland", "Pacific/Fiji",
    "America/Argentina", "America/Sao_Paulo", "America/Santiago", "America/Montevideo",
    "America/La_Paz", "America/Asuncion", "America/Lima", "Africa/Johannesburg",
    "Africa/Nairobi", "Africa/Harare", "Indian/",
)
_southern_zones: Optional[set[str]] = None


def _load_southern_zones() -> Optional[set[str]]:
    """Zones south of the equator, read from the tz database's coordinates
    (e.g. "AU\t-3352+15113\tAustralia/Sydney") so this isn't a list someone
    has to keep up to date."""
    for path in _ZONE_TAB_PATHS:
        try:
            with open(path) as f:
                zones = set()
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    coords, name = parts[1], parts[2].strip()
                    if coords[:1] == "-":  # latitude sign
                        zones.add(name)
                if zones:
                    return zones
        except OSError:
            continue
    return None


def is_southern_hemisphere(timezone: Optional[str]) -> bool:
    global _southern_zones
    if not timezone:
        return False
    if _southern_zones is None:
        _southern_zones = _load_southern_zones() or set()
    if _southern_zones:
        return timezone in _southern_zones
    return timezone.startswith(_SOUTHERN_PREFIXES)


def _astronomical_season(today: datetime.date) -> str:
    season = "winter"
    for month, day, name in _ASTRONOMICAL_STARTS:
        if (today.month, today.day) >= (month, day):
            season = name
    return season


def accepted_seasons(today: datetime.date, timezone: Optional[str] = None) -> set[str]:
    """Seasons it's reasonable to call today.

    Both the meteorological season (by month) and the astronomical one (by
    equinox/solstice) are accepted: for two or three weeks each quarter they
    disagree — on 15 September a patient saying "summer" is as right as one
    saying "fall" — and marking that wrong flags a healthy patient.
    Flipped below the equator.
    """
    seasons = {_SEASON_BY_MONTH[today.month], _astronomical_season(today)}
    if is_southern_hemisphere(timezone):
        seasons = {_OPPOSITE_SEASON[s] for s in seasons}
    return seasons


def patient_now(timezone: Optional[str] = None) -> datetime.datetime:
    """The current time where the patient is, not where the server is."""
    if timezone:
        try:
            return datetime.datetime.now(ZoneInfo(timezone))
        except Exception:
            pass  # unknown zone: fall back to the server clock below
    return datetime.datetime.now()


def orientation_expected_answer(
    question_id: str,
    today: Optional[datetime.date] = None,
    timezone: Optional[str] = None,
) -> Optional[set[str]]:
    """Accepted answers for the orientation prompts in RECALL_CHECK, or None
    for prompts that aren't orientation questions.

    Answers are judged in the patient's own timezone. The server clock was
    wrong for anyone in a different one: a New York patient called at 8pm is
    already on the next day in the UK, so "what day is it?" was scored wrong.
    """
    today = today or patient_now(timezone).date()
    if question_id == "recall_check:0":
        return {today.strftime("%A")}
    if question_id == "recall_check:1":
        return accepted_seasons(today, timezone)
    return None


@dataclass
class CallScript:
    state: CallState = CallState.RECALL_CHECK
    prompt_index: int = 0
    turn_count: int = 0
    # The words this call asks the patient to remember (see recall_words_for_call).
    recall_words: tuple[str, ...] = RECALL_WORD_LISTS[0]

    def current_prompt(self) -> Optional[str]:
        """The prompt to say right now, or None if the call is complete."""
        prompts = STATE_PROMPTS[self.state]
        if self.prompt_index >= len(prompts):
            return None
        return prompts[self.prompt_index].format(words=", ".join(self.recall_words))

    def current_question_id(self) -> str:
        return f"{self.state.name.lower()}:{self.prompt_index}"

    def is_complete(self) -> bool:
        return self.state == CallState.COMPLETE

    def capture_mode(self) -> CaptureMode:
        return CAPTURE_MODES.get(self.state, CaptureMode.ANSWER)

    def segment_name(self) -> Optional[str]:
        return SEGMENT_NAMES.get(self.state)

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
