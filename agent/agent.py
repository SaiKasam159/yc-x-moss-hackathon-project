"""
LiveKit voice agent — Zone A.

Pipeline: STT (Deepgram, word-level timestamps) -> trigger check -> Moss
retrieval (only if triggered) -> LLM -> TTS (edge-tts today; Kokoro-82M
stubbed, see _synthesize_kokoro).

Uses low-level LiveKit Agents primitives (JobContext + manual STT stream +
manual AudioSource publish), not the high-level VoicePipelineAgent wrapper,
because a trigger firing needs to inject retrieved Moss context into the LLM
call before it runs — the high-level wrapper doesn't expose that seam.

Run `python -m agent.agent --dry-run` to exercise the turn logic with no
external services, or `python -m agent.agent start` to run the live worker
(needs LIVEKIT_*/DEEPGRAM_API_KEY in .env). The live path has been run
against LiveKit Cloud end-to-end: STT with word timestamps, trigger checks,
Moss retrieval, the dead-air bridging phrase, TTS playback, and clean
shutdown with audio + transcript persisted.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import sqlite3
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

from agent.call_script import CallScript, CallState, orientation_expected_answer
from agent.triggers import TriggerResult, check_symptom_flag, check_wrong_answer
from db.contracts import MossQARecord
from db.moss_client import ingest_qa_record, push_patient_session, query_patient_history

logger = logging.getLogger("agent")

# --- env / config ----------------------------------------------------------

LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Kokoro tried first per spec; defaults to edge-tts because Kokoro isn't
# wired yet (see _synthesize_kokoro for why) — flip this once it is.
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "edge-tts")  # "kokoro" | "edge-tts"
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-AriaNeural")

SAMPLE_RATE = 16000
NUM_CHANNELS = 1

CALLS_DIR = Path(os.environ.get("CALLS_DIR", "calls"))
DB_PATH = os.environ.get("DB_PATH", "data/app.db")

# No dead air: if retrieval + LLM follow-up generation takes longer than
# this, speak the bridging phrase first, then the real follow-up once ready.
FALLBACK_TIMEOUT_S = float(os.environ.get("FALLBACK_TIMEOUT_S", "1.2"))
FALLBACK_PHRASE = "Can you tell me a bit more about that?"

# Most follow-ups to ask about any single question before moving on, so a
# repeatedly-tripped trigger can't trap the call on one question.
MAX_FOLLOWUPS_PER_QUESTION = int(os.environ.get("MAX_FOLLOWUPS_PER_QUESTION", "2"))

# How long to wait for a final transcript before moving the call on anyway.
# Essential, not just defensive: the SUSTAINED_PHONATION task asks for "ahhh",
# which is not speech and which STT may never emit a final transcript for, and
# a patient may simply stay silent. Without this the call waits forever.
NO_ANSWER_TIMEOUT_S = float(os.environ.get("NO_ANSWER_TIMEOUT_S", "12"))

# Who gets called and under which call id is now resolved at runtime — see
# resolve_patient() and start_call_row(). PATIENT_ID pins a specific patient;
# otherwise the most recent signup is called.
FALLBACK_PATIENT_NAME = "Test Patient"  # only used if the patients table is empty
# Stands in for a real baseline-call lookup for the personal-recall question.
BASELINE_BREAKFAST_ANSWER = os.environ.get("BASELINE_BREAKFAST_ANSWER", "eggs")


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# --- turn context / handling ------------------------------------------------

@dataclass
class TurnContext:
    patient_id: int
    call_id: int
    # question_id -> known-correct answer text, for RECALL_CHECK prompts
    # whose expected answer isn't computed dynamically (i.e. personal
    # recall, not orientation — see call_script.orientation_expected_answer).
    expected_answers: dict[str, str] = field(default_factory=dict)
    # question_id -> how many follow-ups we've already asked on it. A fired
    # trigger deliberately does NOT advance the script (so the patient can
    # answer the follow-up), which means a patient who keeps tripping the
    # check would loop on one question forever. See MAX_FOLLOWUPS_PER_QUESTION.
    followup_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class TurnOutcome:
    reply_text: str
    trigger: TriggerResult
    used_fallback: bool = False


def _resolve_expected_answer(script: CallScript, ctx: TurnContext) -> Optional[str]:
    qid = script.current_question_id()
    dynamic = orientation_expected_answer(qid)
    if dynamic is not None:
        return dynamic
    return ctx.expected_answers.get(qid)


async def _retrieve_and_generate_followup(
    script: CallScript, ctx: TurnContext, record: MossQARecord, trigger: TriggerResult
) -> str:
    """The Moss retrieval + LLM follow-up step that runs when a trigger
    fires. moss_client's real SessionIndex.query() runs in-memory (no
    network round trip) so this stays fast enough for mid-call latency."""
    context_records = await query_patient_history(
        patient_id=ctx.patient_id,
        query_text=trigger.query_text or record.answer_text,
        question_topic=record.question_topic,
    )
    return await generate_followup_llm(trigger, context_records)


async def speak_with_fallback(
    generate: Awaitable[str], speak: Callable[[str], Awaitable[None]]
) -> str:
    """Race `generate` against FALLBACK_TIMEOUT_S. If it's not ready in
    time, speak the bridging phrase first (so there's never dead air), then
    speak the real result once it lands. Returns the text actually used as
    the final reply.
    """
    task = asyncio.ensure_future(generate)
    try:
        result = await asyncio.wait_for(asyncio.shield(task), timeout=FALLBACK_TIMEOUT_S)
        return result
    except asyncio.TimeoutError:
        await speak(FALLBACK_PHRASE)
        result = await task  # already in flight, just await completion
        return result


async def handle_turn(
    script: CallScript,
    ctx: TurnContext,
    answer_text: str,
    speak: Callable[[str], Awaitable[None]],
    word_timestamps: Optional[list] = None,
) -> TurnOutcome:
    """Given one captured answer: ingest it into Moss, run the relevant
    trigger check for the current state, and return what the agent should
    say next — either an informed follow-up (trigger fired) or the next
    scripted prompt (trigger didn't fire / call complete).

    `speak` is called directly (not just returned) when the fallback phrase
    is needed mid-generation, so the caller doesn't need to know that
    happened to avoid dead air.
    """
    record = MossQARecord(
        patient_id=ctx.patient_id,
        call_id=ctx.call_id,
        question_id=script.current_question_id(),
        question_topic=script.state.name.lower(),
        answer_text=answer_text,
        timestamp=_now_iso(),
        extra={"word_timestamps": word_timestamps} if word_timestamps else {},
    )
    await ingest_qa_record(record)

    trigger: TriggerResult = TriggerResult(fired=False)
    if script.state == CallState.RECALL_CHECK:
        trigger = check_wrong_answer(answer_text, _resolve_expected_answer(script, ctx))
    elif script.state == CallState.OPEN_QA:
        trigger = check_symptom_flag(answer_text)

    qid = record.question_id
    already_followed_up = ctx.followup_counts.get(qid, 0)

    if trigger.fired and already_followed_up >= MAX_FOLLOWUPS_PER_QUESTION:
        # Don't ask a third time about the same question — move the call on.
        # Without this a patient who keeps tripping the check (or whose
        # answer keeps mis-transcribing) loops on one question forever and
        # the call never reaches the later tasks.
        logger.info(
            "trigger fired on %s but follow-up limit (%d) reached — advancing",
            qid, MAX_FOLLOWUPS_PER_QUESTION,
        )
    elif trigger.fired:
        ctx.followup_counts[qid] = already_followed_up + 1
        logger.info("trigger fired on %s: %s", qid, trigger.reason)
        generate = _retrieve_and_generate_followup(script, ctx, record, trigger)
        before = asyncio.get_event_loop().time()
        reply = await speak_with_fallback(generate, speak)
        used_fallback = (asyncio.get_event_loop().time() - before) >= FALLBACK_TIMEOUT_S
        return TurnOutcome(reply_text=reply, trigger=trigger, used_fallback=used_fallback)

    next_prompt = script.advance()
    if script.is_complete():
        return TurnOutcome(reply_text="That's everything for today — thanks for checking in!", trigger=trigger)
    return TurnOutcome(reply_text=next_prompt, trigger=trigger)


# --- LLM follow-up generation ------------------------------------------------

_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import AsyncOpenAI  # local import: optional dep until an LLM key exists
        _llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _llm_client


async def generate_followup_llm(trigger: TriggerResult, context_records: list[MossQARecord]) -> str:
    """Generate one short, informed follow-up question using retrieved Moss
    context. Degrades to a templated question (no network call) when
    OPENAI_API_KEY isn't set, so the whole loop stays runnable without an
    LLM key — same spirit as db/moss_client.py's STUB_MODE.
    """
    context_lines = [f"- ({r.timestamp}) {r.answer_text}" for r in context_records]
    context_str = "\n".join(context_lines) or "(no related prior answers on file)"

    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — using templated follow-up instead of a real LLM call.")
        return "Can you say more about that? I want to make sure I understand what you meant."

    system = (
        "You are a warm, brief phone check-in assistant for a Parkinson's "
        f"patient. A trigger fired during the call: {trigger.reason}. "
        "Using the retrieved context below, ask ONE short, natural, "
        "non-alarming follow-up question. Do not diagnose or give medical "
        "advice. One sentence only."
    )
    try:
        client = _get_llm_client()
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=100,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Retrieved context:\n{context_str}\n\nGenerate the follow-up question.",
                },
            ],
        )
        return (response.choices[0].message.content or "").strip() or FALLBACK_PHRASE
    except Exception:
        # Never let an LLM outage drop the call — fall back to a safe line.
        logger.warning("LLM follow-up generation failed; using templated follow-up", exc_info=True)
        return "Can you say more about that? I want to make sure I understand what you meant."


# --- TTS ---------------------------------------------------------------------

async def synthesize_speech(text: str) -> bytes:
    """Returns PCM16 mono audio at SAMPLE_RATE for `text`, via whichever
    provider TTS_PROVIDER selects."""
    if TTS_PROVIDER == "edge-tts":
        return await _synthesize_edge_tts(text)
    if TTS_PROVIDER == "kokoro":
        return await _synthesize_kokoro(text)
    raise ValueError(f"Unknown TTS_PROVIDER: {TTS_PROVIDER!r}")


async def _synthesize_edge_tts(text: str) -> bytes:
    """edge-tts: free, no API key, works today. Streams mp3, decoded to
    PCM16 via PyAV (pip-installable, no system ffmpeg/espeak-ng needed)."""
    import edge_tts

    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    mp3_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes.extend(chunk["data"])
    # Decode off the event loop: PyAV is synchronous CPU work, and LiveKit
    # warns ("event loop blocked for 1280ms") that blocking here delays
    # outgoing audio and turn handling for the whole call.
    return await asyncio.to_thread(_decode_to_pcm16, bytes(mp3_bytes))


async def _synthesize_kokoro(text: str) -> bytes:
    """Kokoro-82M self-hosted TTS — tried first per spec, NOT wired yet.

    Deferred because getting it running needs, beyond a pip install:
      - the `espeak-ng` phonemizer backend, a system-level package
        (`brew install espeak-ng` on macOS) — exactly the kind of install
        this project asks to confirm before running
      - ~327MB of model weights to download
      - likely `torch` as a dependency (heavier install than edge-tts)

    That's more setup than fits a "get one round trip proven" first step,
    so TTS_PROVIDER defaults to edge-tts for now. To switch: confirm the
    espeak-ng install, `pip install kokoro`, set TTS_PROVIDER=kokoro and
    KOKORO_MODEL_PATH in .env, and implement synthesis here (e.g. via the
    `kokoro` package's KPipeline, matching the return contract above:
    PCM16 mono bytes at SAMPLE_RATE).
    """
    raise NotImplementedError("Kokoro TTS not wired yet — see docstring. Using TTS_PROVIDER=edge-tts instead.")


def _decode_to_pcm16(compressed_audio: bytes) -> bytes:
    """Decode mp3 (or anything PyAV's ffmpeg build reads) to raw PCM16 mono
    at SAMPLE_RATE, for LiveKit AudioFrame publishing."""
    import av

    container = av.open(io.BytesIO(compressed_audio))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pcm = bytearray()
    stream = container.streams.audio[0]
    for packet in container.demux(stream):
        for frame in packet.decode():
            for resampled in resampler.resample(frame):
                pcm.extend(bytes(resampled.planes[0]))
    return bytes(pcm)


# --- call recording (audio + transcript with word-level timestamps) --------

class CallRecorder:
    """Writes call audio to <CALLS_DIR>/<call_id>/audio.wav and a transcript
    (with Deepgram word-level timestamps on patient turns) to
    <CALLS_DIR>/<call_id>/transcript.json, then records both paths into the
    `calls` table per db/schema.sql (audio_path, transcript_path) — without
    modifying db/schema.sql or db/contracts.py.
    """

    def __init__(self, call_id: int, patient_id: int, calls_dir: Path = CALLS_DIR):
        self.call_id = call_id
        self.patient_id = patient_id
        self.call_dir = calls_dir / str(call_id)
        self.call_dir.mkdir(parents=True, exist_ok=True)
        self.audio_path = self.call_dir / "audio.wav"
        self.transcript_path = self.call_dir / "transcript.json"
        self.turns: list[dict] = []

        self._wav = wave.open(str(self.audio_path), "wb")
        self._wav.setnchannels(NUM_CHANNELS)
        self._wav.setsampwidth(2)  # PCM16
        self._wav.setframerate(SAMPLE_RATE)

    def write_audio_frame(self, pcm16_bytes: bytes) -> None:
        self._wav.writeframes(pcm16_bytes)

    def log_patient_turn(self, text: str, word_timestamps: Optional[list] = None) -> None:
        self.turns.append({
            "speaker": "patient",
            "text": text,
            "words": word_timestamps or [],
            "timestamp": _now_iso(),
        })

    def log_agent_turn(self, text: str) -> None:
        self.turns.append({"speaker": "agent", "text": text, "timestamp": _now_iso()})

    def finalize(self, db_path: str = DB_PATH) -> None:
        self._wav.close()
        with open(self.transcript_path, "w") as f:
            json.dump(
                {"call_id": self.call_id, "patient_id": self.patient_id, "turns": self.turns},
                f, indent=2,
            )
        _record_call_row(db_path, self.call_id, self.patient_id, str(self.audio_path), str(self.transcript_path))
        logger.info("call %s recorded: %s, %s", self.call_id, self.audio_path, self.transcript_path)


def _connect_db(db_path: str) -> sqlite3.Connection:
    """Open the call database, applying db/schema.sql (read-only use — this
    never modifies the shared schema file)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    schema_sql = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql.read_text())
    return conn


def resolve_patient(db_path: str = DB_PATH) -> tuple[int, str]:
    """Decide who this call is for.

    PATIENT_ID wins if set; otherwise the most recently signed-up patient
    (so a patient registered through /signup actually gets called); otherwise
    a test patient is created so the agent still runs on an empty database.
    """
    conn = _connect_db(db_path)
    try:
        forced = os.environ.get("PATIENT_ID")
        if forced:
            row = conn.execute("SELECT id, name FROM patients WHERE id = ?", (int(forced),)).fetchone()
            if row:
                return int(row[0]), row[1]
            logger.warning("PATIENT_ID=%s not found in %s; falling back", forced, db_path)

        row = conn.execute("SELECT id, name FROM patients ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            return int(row[0]), row[1]

        cur = conn.execute("INSERT INTO patients (name) VALUES (?)", (FALLBACK_PATIENT_NAME,))
        conn.commit()
        return int(cur.lastrowid), FALLBACK_PATIENT_NAME
    finally:
        conn.close()


def start_call_row(patient_id: int, db_path: str = DB_PATH) -> int:
    """Create this call's row up front and return its id.

    Allocating a fresh id per call (instead of a hardcoded one) is what keeps
    repeat calls from overwriting each other's audio, transcript and row.
    """
    conn = _connect_db(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO calls (patient_id, timestamp) VALUES (?, ?)",
            (patient_id, _now_iso()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _record_call_row(db_path: str, call_id: int, patient_id: int, audio_path: str, transcript_path: str) -> None:
    """Attach the recorded artefacts to the call row created at call start."""
    conn = _connect_db(db_path)
    try:
        conn.execute(
            "UPDATE calls SET audio_path = ?, transcript_path = ? WHERE id = ?",
            (audio_path, transcript_path, call_id),
        )
        conn.commit()
    finally:
        conn.close()


# --- dry run (no external keys needed) --------------------------------------

async def dry_run() -> None:
    """Exercises the full turn-handling loop — call script, both triggers,
    stub Moss retrieval, fallback bridging, LLM step (templated if
    ANTHROPIC_API_KEY is unset) — with scripted fake patient answers.
    Zero LiveKit/Deepgram/Moss/LLM keys required. This is the "prove the
    pipeline works" check that's actually runnable in this environment.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    patient_id, patient_name = resolve_patient()
    call_id = start_call_row(patient_id)
    print(f"[dry-run] patient {patient_id} ({patient_name}), call {call_id}\n")

    script = CallScript()
    ctx = TurnContext(
        patient_id=patient_id,
        call_id=call_id,
        expected_answers={"recall_check:2": BASELINE_BREAKFAST_ANSWER},
    )
    recorder = CallRecorder(call_id, patient_id)

    async def speak(text: str) -> None:
        print(f"AGENT: {text}")
        recorder.log_agent_turn(text)

    # Scripted answers per question_id, as a queue: a triggering first answer
    # (wrong recall / symptom mention) followed by a clean follow-up answer,
    # so the follow-up round the agent asks actually gets resolved instead
    # of re-triggering on the same wrong answer forever. Everything else is
    # a single plausible "correct" answer.
    scripted_answers: dict[str, list[str]] = {
        "recall_check:0": [datetime.date.today().strftime("%A")],          # correct — no trigger
        "recall_check:1": ["definitely not a season", "sorry, I meant fall"],   # WRONG, then corrected — triggers once
        "recall_check:2": ["I had toast, not eggs", "oh you're right, I had eggs"],  # WRONG vs baseline — triggers once
        "sustained_phonation:0": ["ahhhhh"],
        "reading_task:0": ["The old dog stretched slowly..."],
        "open_qa:0": ["I've been okay, but my hands have had a bit of a tremor lately.", "it's mild, comes and goes"],  # triggers symptom flag once
        "open_qa:1": ["Nothing major otherwise."],
        "counting_task:0": ["1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"],
    }
    answer_iters = {qid: iter(answers) for qid, answers in scripted_answers.items()}

    def next_answer(qid: str) -> str:
        try:
            return next(answer_iters.get(qid, iter(())))
        except StopIteration:
            return "that's all, thank you"  # never re-triggers — safe default once a qid's queue is exhausted

    await speak(script.current_prompt())
    max_turns = 30  # safety net against any future scripting bug causing an infinite loop
    for _ in range(max_turns):
        if script.is_complete():
            break
        qid = script.current_question_id()
        answer = next_answer(qid)
        print(f"PATIENT: {answer}")
        recorder.log_patient_turn(answer)

        outcome = await handle_turn(script, ctx, answer, speak=speak)
        await speak(outcome.reply_text)
        if outcome.trigger.fired:
            print(f"  (trigger: {outcome.trigger.reason} | fallback used: {outcome.used_fallback})")
    else:
        print(f"[dry-run] hit max_turns={max_turns} safety cap without completing — check for a trigger loop.")

    recorder.finalize()
    await push_patient_session(ctx.patient_id)  # persist this patient's Moss session for their next call
    print(f"\nTranscript: {recorder.transcript_path}")
    print(f"Audio (silent placeholder in dry-run — no real TTS/mic captured): {recorder.audio_path}")


# --- live LiveKit entrypoint --------------------------------------------------
# NOT YET RUNNABLE without LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET and
# DEEPGRAM_API_KEY in .env — see module docstring. Written against the
# current livekit-agents / livekit-plugins-deepgram APIs; verify field/method
# names against your installed versions once those keys are in place.

async def _forward_audio(track, stt_stream) -> None:
    from livekit import rtc
    audio_stream = rtc.AudioStream(track)
    async for event in audio_stream:
        stt_stream.push_frame(event.frame)


async def _play_pcm16(audio_source, pcm16_bytes: bytes, recorder: Optional[CallRecorder] = None) -> None:
    from livekit import rtc

    frame_ms = 20
    bytes_per_frame = int(SAMPLE_RATE * frame_ms / 1000) * 2  # PCM16
    for i in range(0, len(pcm16_bytes), bytes_per_frame):
        chunk = pcm16_bytes[i:i + bytes_per_frame]
        if not chunk:
            continue
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=len(chunk) // 2,
        )
        await audio_source.capture_frame(frame)
    if recorder is not None:
        recorder.write_audio_frame(pcm16_bytes)


def _create_stt():
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY not set — see .env.example")
    from livekit.plugins import deepgram

    # Deepgram returns word-level timestamps by default (the `words` array
    # on each alternative); consumed below via alt.words.
    return deepgram.STT(
        api_key=DEEPGRAM_API_KEY,
        model="nova-2",
        language="en-US",
        punctuate=True,
        smart_format=True,
        interim_results=True,
    )


async def entrypoint(ctx) -> None:
    from livekit import rtc
    from livekit.agents import AutoSubscribe
    from livekit.agents import stt as lk_stt

    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise RuntimeError("LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET not set — see .env.example")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await ctx.wait_for_participant()

    # Resolved per call, off the main loop (sqlite is blocking): who we're
    # calling, and a fresh call id so repeat calls don't overwrite each other.
    patient_id, patient_name = await asyncio.to_thread(resolve_patient)
    call_id = await asyncio.to_thread(start_call_row, patient_id)
    logger.info("starting call %s for patient %s (%s)", call_id, patient_id, patient_name)

    script = CallScript()
    turn_ctx = TurnContext(
        patient_id=patient_id,
        call_id=call_id,
        expected_answers={"recall_check:2": BASELINE_BREAKFAST_ANSWER},
    )
    recorder = CallRecorder(call_id, patient_id)

    audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("agent-voice", audio_source)
    await ctx.room.local_participant.publish_track(track)

    async def speak(text: str) -> None:
        pcm = await synthesize_speech(text)
        await _play_pcm16(audio_source, pcm, recorder=recorder)
        recorder.log_agent_turn(text)

    stt_stream = _create_stt().stream()

    # Without this the job never ends when the patient hangs up: the STT
    # stream just blocks, and LiveKit eventually force-cancels the entrypoint
    # ("entrypoint did not exit in time"), so the call shuts down dirtily.
    disconnected = asyncio.Event()

    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(*_args):
        disconnected.set()

    @ctx.room.on("disconnected")
    def _on_room_disconnected(*_args):
        disconnected.set()

    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(track_, *_args):
        if track_.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(_forward_audio(track_, stt_stream))

    await speak(script.current_prompt())

    # The whole conversation runs under try/finally: a call that ends early —
    # patient hangs up, network drops, the room closes — must still write its
    # transcript and persist to Moss. Without this, anything short of a fully
    # completed 5-stage call loses the transcript entirely (the audio survives,
    # since wave writes incrementally, but every word and timestamp is gone),
    # which is exactly the data the ML pipeline needs.
    stt_iter = stt_stream.__aiter__()
    try:
        while not script.is_complete():
            if disconnected.is_set():
                logger.info("participant left — ending call %s", turn_ctx.call_id)
                break

            try:
                event = await asyncio.wait_for(
                    stt_iter.__anext__(), timeout=NO_ANSWER_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                # No transcript arrived. Either a non-speech task (the "ahhh"
                # phonation stage never produces one) or a silent patient —
                # either way, keep the call moving instead of hanging.
                logger.info(
                    "no transcript within %ss on %s — advancing",
                    NO_ANSWER_TIMEOUT_S, script.current_question_id(),
                )
                next_prompt = script.advance()
                if script.is_complete():
                    await speak("That's everything for today — thanks for checking in!")
                    break
                await speak(next_prompt)
                continue
            except StopAsyncIteration:
                break

            if event.type != lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
                continue
            alt = event.alternatives[0]
            answer_text = alt.text
            # livekit-agents >=1.x gives words as TimedString: a str subclass
            # carrying start_time/end_time. There is no `.word` attribute — the
            # word itself IS the string, so str(w) is the text.
            word_timestamps = [
                {
                    "word": str(w),
                    "start": getattr(w, "start_time", None),
                    "end": getattr(w, "end_time", None),
                }
                for w in (getattr(alt, "words", None) or [])
            ]
            recorder.log_patient_turn(answer_text, word_timestamps)

            outcome = await handle_turn(
                script, turn_ctx, answer_text, speak=speak, word_timestamps=word_timestamps
            )
            await speak(outcome.reply_text)

            if script.is_complete():
                break
    finally:
        try:
            recorder.finalize()
        except Exception:
            logger.exception("failed to finalize call recording for call %s", turn_ctx.call_id)
        try:
            # Persist this patient's Moss session for their next call.
            await push_patient_session(turn_ctx.patient_id)
        except Exception:
            logger.exception("failed to push Moss session for patient %s", turn_ctx.patient_id)


def run_worker() -> None:
    from livekit.agents import WorkerOptions, cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv or len(sys.argv) == 1:
        asyncio.run(dry_run())
    else:
        run_worker()
