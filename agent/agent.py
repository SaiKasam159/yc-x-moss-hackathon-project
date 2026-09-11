"""
LiveKit voice agent — Zone A.

Pipeline: STT (Deepgram, word-level timestamps) -> trigger check -> Moss
retrieval (only if triggered) -> LLM -> TTS (Kokoro-82M, fallback edge-tts).

This needs low-level control between STT and LLM (not the high-level
VoicePipelineAgent convenience wrapper) so a trigger firing can inject
retrieved Moss context into the LLM call before it runs.

STUB: no real LiveKit session wiring yet. This lays out the shape of the
loop and where each piece plugs in. Requires LIVEKIT_URL/API_KEY/API_SECRET
and DEEPGRAM_API_KEY (see .env.example) once implemented for real.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from agent.call_script import CallScript, CallState
from agent.triggers import check_wrong_answer, check_symptom_flag, TriggerResult
from db.contracts import MossQARecord
from db.moss_client import ingest_qa_record, query_patient_history

LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "kokoro")  # "kokoro" | "edge-tts"

# TODO(agent): pick and wire an LLM provider for response generation.
# Not yet decided — see README "Open questions". Needs its own env var
# (e.g. LLM_API_KEY) once chosen.


@dataclass
class TurnContext:
    patient_id: int
    call_id: int
    expected_answer: Optional[str] = None  # for RECALL_CHECK, pulled from baseline call


def handle_turn(script: CallScript, ctx: TurnContext, answer_text: str) -> str:
    """Given one captured answer, decide whether to retrieve + follow up, or
    advance to the next scripted state. Returns the text the agent should say
    next (a follow-up question, or the next scripted prompt).

    This is the STT -> [trigger check] -> [Moss retrieval] -> LLM boundary
    that needs low-level LiveKit Agents control to intercept.
    """
    record = MossQARecord(
        patient_id=ctx.patient_id,
        call_id=ctx.call_id,
        question_id=script.current_question_id(),
        question_topic=script.state.name.lower(),
        answer_text=answer_text,
        timestamp=__import__("datetime").datetime.utcnow().isoformat() + "Z",
    )
    ingest_qa_record(record)

    trigger: TriggerResult = TriggerResult(fired=False)
    if script.state == CallState.RECALL_CHECK:
        trigger = check_wrong_answer(answer_text, ctx.expected_answer)
    elif script.state == CallState.OPEN_QA:
        trigger = check_symptom_flag(answer_text)

    if trigger.fired:
        context_records = query_patient_history(
            patient_id=ctx.patient_id,
            query_text=trigger.query_text or answer_text,
            question_topic=record.question_topic,
        )
        # TODO(agent): feed `context_records` + `trigger.reason` into the LLM
        # call to generate an informed follow-up question, instead of this
        # placeholder.
        return f"[TODO(agent): LLM follow-up using {len(context_records)} retrieved records — {trigger.reason}]"

    script.advance()
    if script.is_complete():
        return "That's everything for today — thanks for checking in!"
    return script.current_prompts()[0]


def run_call(patient_id: int, call_id: int, expected_answer: Optional[str] = None) -> None:
    """Entry point for a live call session. STUB — no real LiveKit room /
    STT / TTS wiring yet.

    TODO(agent):
      - connect to LiveKit room (LIVEKIT_URL/API_KEY/API_SECRET)
      - stream mic audio -> Deepgram STT (word-level timestamps on, for
        ml/audio.py pause detection downstream)
      - on each finalized STT turn, call handle_turn(...)
      - synthesize the returned text via Kokoro-82M (fall back to edge-tts)
        and play it into the room
      - loop until script.is_complete()
    """
    script = CallScript()
    ctx = TurnContext(patient_id=patient_id, call_id=call_id, expected_answer=expected_answer)
    print(f"[stub] would start call {call_id} for patient {patient_id}")
    print(f"[stub] first prompt: {script.current_prompts()[0]}")
    raise NotImplementedError("Real LiveKit session loop not wired yet.")


if __name__ == "__main__":
    # Local dry run of the turn-handling logic only (no real call).
    script = CallScript()
    ctx = TurnContext(patient_id=1, call_id=1, expected_answer="eggs")
    print("prompt:", script.current_prompts()[0])
    next_prompt = handle_turn(script, ctx, "I had toast this morning")
    print("agent says:", next_prompt)
