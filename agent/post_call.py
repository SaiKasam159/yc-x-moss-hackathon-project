"""
After-call analysis — Zone A orchestration over the ML pipeline.

Runs when a call ends: takes the audio segments the call recorded, extracts
features, predicts a UPDRS score, writes a report, and stores the results in
the `features` and `updrs_predictions` tables.

Nothing connected these before. A finished call left a transcript and a WAV
and stopped there, so everything in ml/ and reports/ only ever ran by hand —
no call ever produced a score or a report.

It runs as its own process (the agent spawns it detached) for two reasons:
the work is heavy CPU that would stall a live call, and it has to outlive the
LiveKit job, which is torn down as soon as the call ends.

Re-runnable by hand, which is also how to analyse calls recorded earlier:

    python -m agent.post_call --call-id 12
    python -m agent.post_call --latest
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import math
import os
import sqlite3
from pathlib import Path
from typing import Optional

from db.contracts import FeatureVector

logger = logging.getLogger("agent.post_call")

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("DB_PATH", "data/app.db")
CALLS_DIR = Path(os.environ.get("CALLS_DIR", "calls"))

# Shorter than this and the vowel isn't worth measuring.
MIN_PHONATION_SECONDS = float(os.environ.get("MIN_PHONATION_SECONDS", "1.0"))


def _connect(db_path: str) -> sqlite3.Connection:
    """Open the call database, applying db/schema.sql (read-only use of it)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript((REPO_ROOT / "db" / "schema.sql").read_text())
    return conn


def _load_transcript(call_id: int, calls_dir: Path) -> dict:
    path = calls_dir / str(call_id) / "transcript.json"
    if not path.exists():
        raise FileNotFoundError(f"No transcript for call {call_id} at {path}")
    return json.loads(path.read_text())


def _segment(transcript: dict, name: str) -> Optional[dict]:
    """A recorded segment by name, if it has usable audio."""
    for seg in transcript.get("segments", []):
        if seg.get("name") == name and seg.get("path") and Path(seg["path"]).exists():
            return seg
    return None


def _acoustic_features(
    transcript: dict, calls_dir: Path, call_id: int, allow_speech_fallback: bool = False,
) -> tuple[Optional[dict], Optional[str]]:
    """Jitter/shimmer/HNR/RPDE/DFA/PPE from the sustained vowel.

    Returns (features, reason_unusable). These measurements only mean anything
    on a sustained vowel, so a call without one gets no features rather than
    numbers taken from conversational speech.
    """
    from ml.audio import extract_acoustic_features, extract_phonation_segment

    seg = _segment(transcript, "phonation")
    if seg:
        if (seg.get("duration_s") or 0) < MIN_PHONATION_SECONDS:
            return None, (f"phonation segment is only {seg.get('duration_s')}s "
                          f"(need {MIN_PHONATION_SECONDS}s)")
        return extract_acoustic_features(seg["path"]), None

    if not allow_speech_fallback:
        # Measuring ordinary speech as if it were a held vowel produced
        # nonsense on a hung-up call: jitter 4.6%, HNR 7.7 dB, "UPDRS 29.3".
        return None, "no phonation segment was recorded (patient didn't hold a vowel)"

    patient_wav = transcript.get("audio", {}).get("patient") or str(calls_dir / str(call_id) / "patient.wav")
    logger.warning("call %s: no phonation segment; falling back to VAD over %s "
                   "— these features come from speech, not a held vowel", call_id, patient_wav)
    return extract_acoustic_features(extract_phonation_segment(patient_wav)), None


def _unusable_reason(acoustic: dict) -> Optional[str]:
    """Why these numbers can't be trusted, if they can't.

    Silence makes parselmouth return NaN for jitter and shimmer; those NaNs
    were being written to the database and scored as "high severity".
    """
    bad = [k for k, v in acoustic.items() if v is None or not math.isfinite(v)]
    if bad:
        return f"no usable voice sample (these came back as not-a-number: {', '.join(sorted(bad))})"
    if acoustic.get("hnr", 0) <= 0:
        return (f"no harmonic signal in the recording (HNR {acoustic['hnr']:.1f} dB) — "
                "silence or noise, not a voice")
    return None


def _prosody_features(transcript: dict) -> dict:
    """Speech rate and pauses, preferring the reading passage over counting."""
    from ml.audio import extract_prosody_features

    for name in ("reading", "counting"):
        seg = _segment(transcript, name)
        if seg and seg.get("words"):
            return extract_prosody_features(seg["path"], seg["words"])
    logger.warning("no reading or counting segment with words; prosody features will be zero")
    return {"speech_rate": 0.0, "pause_freq": 0.0, "pause_avg_duration": 0.0}


def _ensure_call_row(conn: sqlite3.Connection, call_id: int, patient_id: int) -> None:
    """Make sure the call and its patient exist before writing results.

    The agent creates both when a call starts. Analysing a call whose rows
    aren't there — a calls/ directory copied to another machine, a rebuilt
    database — used to fail with a bare "FOREIGN KEY constraint failed"
    halfway through. The transcript knows who the call belonged to, so
    rebuild the rows from it and say so.
    """
    if conn.execute("SELECT 1 FROM calls WHERE id = ?", (call_id,)).fetchone():
        return
    conn.execute("INSERT OR IGNORE INTO patients (id, name) VALUES (?, ?)",
                 (patient_id, f"Patient {patient_id}"))
    conn.execute("INSERT INTO calls (id, patient_id, timestamp) VALUES (?, ?, ?)",
                 (call_id, patient_id, datetime.datetime.utcnow().isoformat() + "Z"))
    conn.commit()
    logger.warning("call %s had no database row; rebuilt it from the transcript", call_id)


def _save_features(conn: sqlite3.Connection, features: FeatureVector) -> None:
    columns = [f.name for f in dataclasses.fields(FeatureVector)]
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c != "call_id")
    conn.execute(
        f"INSERT INTO features ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))}) "
        f"ON CONFLICT(call_id) DO UPDATE SET {updates}",
        [getattr(features, c) for c in columns],
    )
    conn.commit()


def _save_prediction(conn: sqlite3.Connection, prediction) -> None:
    conn.execute(
        "INSERT INTO updrs_predictions (call_id, predicted_score, confidence_band, anomaly_flag) "
        "VALUES (?, ?, ?, ?) ON CONFLICT(call_id) DO UPDATE SET "
        "predicted_score = excluded.predicted_score, "
        "confidence_band = excluded.confidence_band, "
        "anomaly_flag = excluded.anomaly_flag",
        (prediction.call_id, prediction.predicted_score,
         prediction.confidence_band, int(bool(prediction.anomaly_flag))),
    )
    conn.commit()


def _transcript_text(transcript: dict) -> str:
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in transcript.get("turns", []))


def _write_unusable_report(
    call_id: int, patient_id: int, reason: str, flags: list[str], calls_dir: Path
) -> Path:
    """A call that produced no measurable voice still gets a report saying so."""
    lines = [f"Patient {patient_id} | Call {call_id}"]
    if flags:
        lines.append("Raised during the call:\n  • " + "\n  • ".join(flags))
    lines.append(f"No voice measurements from this call: {reason}.")
    lines.append("No UPDRS estimate is given, because there is nothing to base one on.")
    path = calls_dir / str(call_id) / "report.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def analyse_call(
    call_id: int, db_path: str = DB_PATH, calls_dir: Path = CALLS_DIR,
    allow_speech_fallback: bool = False,
) -> dict:
    """Analyse one finished call. Returns what it produced, for logging."""
    transcript = _load_transcript(call_id, calls_dir)
    patient_id = transcript["patient_id"]
    flags = [f["reason"] for f in transcript.get("flags", []) if f.get("reason")]

    acoustic, unusable = _acoustic_features(transcript, calls_dir, call_id, allow_speech_fallback)
    if acoustic is not None:
        unusable = _unusable_reason(acoustic)

    if unusable:
        # Refuse to invent measurements. Writing NaNs or numbers taken from
        # ordinary speech poisons the patient's history and produced confident
        # "high severity" output from a call where nobody said anything.
        logger.warning("call %s: not scoring this call — %s", call_id, unusable)
        report_path = _write_unusable_report(call_id, patient_id, unusable, flags, calls_dir)
        return {
            "call_id": call_id, "patient_id": patient_id, "scored": False,
            "reason": unusable, "flags": flags, "report_path": str(report_path),
        }

    prosody = _prosody_features(transcript)

    from ml.features import build_feature_vector
    features = build_feature_vector(call_id=call_id, acoustic=acoustic, prosody=prosody)

    conn = _connect(db_path)
    try:
        _ensure_call_row(conn, call_id, patient_id)
        _save_features(conn, features)
        logger.info("call %s: features saved", call_id)

        prediction = None
        try:
            from ml.predict import predict_updrs
            prediction = predict_updrs(features)
            _save_prediction(conn, prediction)
            logger.info("call %s: UPDRS %s (%s)", call_id,
                        prediction.predicted_score, prediction.confidence_band)
        except FileNotFoundError as exc:
            # No trained model yet: features are still worth keeping.
            logger.warning("call %s: no UPDRS prediction — %s", call_id, exc)
    finally:
        conn.close()

    report_path = None
    if prediction is not None:
        from reports.generate_report import generate_report, push_report_to_moss
        report = generate_report(
            patient_id=patient_id, features=features,
            transcript=_transcript_text(transcript), prediction=prediction, flags=flags,
        )
        report_path = calls_dir / str(call_id) / "report.txt"
        report_path.write_text(report.summary_text + "\n")
        logger.info("call %s: report written to %s", call_id, report_path)
        try:
            # Also store it in Moss so a later call can retrieve "last time we
            # noted ..." during the conversation.
            asyncio.run(push_report_to_moss(report))
            logger.info("call %s: report stored in Moss", call_id)
        except Exception:
            logger.warning("call %s: could not store the report in Moss", call_id, exc_info=True)

    return {
        "call_id": call_id,
        "patient_id": patient_id,
        "scored": True,
        "features": dataclasses.asdict(features),
        "predicted_score": prediction.predicted_score if prediction else None,
        "confidence_band": prediction.confidence_band if prediction else None,
        "flags": flags,
        "report_path": str(report_path) if report_path else None,
    }


def _latest_call_id(db_path: str, calls_dir: Path) -> Optional[int]:
    conn = _connect(db_path)
    try:
        for (call_id,) in conn.execute("SELECT id FROM calls ORDER BY id DESC"):
            if (calls_dir / str(call_id) / "transcript.json").exists():
                return int(call_id)
    finally:
        conn.close()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a finished call.")
    parser.add_argument("--call-id", type=int)
    parser.add_argument("--latest", action="store_true", help="analyse the most recent recorded call")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--calls-dir", default=str(CALLS_DIR))
    parser.add_argument(
        "--allow-speech-fallback", action="store_true",
        help="if no phonation segment was recorded, measure ordinary speech instead "
             "(for calls recorded before segments existed; the numbers are not comparable)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    calls_dir = Path(args.calls_dir)

    call_id = args.call_id
    if call_id is None:
        if not args.latest:
            parser.error("pass --call-id N or --latest")
        call_id = _latest_call_id(args.db, calls_dir)
        if call_id is None:
            parser.error("no recorded calls found")

    result = analyse_call(call_id, db_path=args.db, calls_dir=calls_dir,
                          allow_speech_fallback=args.allow_speech_fallback)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
