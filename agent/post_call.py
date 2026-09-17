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
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from db.contracts import FeatureVector

logger = logging.getLogger("agent.post_call")

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("DB_PATH", "data/app.db")
CALLS_DIR = Path(os.environ.get("CALLS_DIR", "calls"))


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


def _acoustic_features(transcript: dict, calls_dir: Path, call_id: int) -> dict:
    """Jitter/shimmer/HNR/RPDE/DFA/PPE from the sustained vowel."""
    from ml.audio import extract_acoustic_features, extract_phonation_segment

    seg = _segment(transcript, "phonation")
    if seg:
        return extract_acoustic_features(seg["path"])

    # No marked segment (e.g. a call recorded before the phonation stage was
    # captured properly): fall back to finding the longest voiced stretch.
    patient_wav = transcript.get("audio", {}).get("patient") or str(calls_dir / str(call_id) / "patient.wav")
    logger.warning("no phonation segment for call %s; falling back to VAD over %s", call_id, patient_wav)
    return extract_acoustic_features(extract_phonation_segment(patient_wav))


def _prosody_features(transcript: dict) -> dict:
    """Speech rate and pauses, preferring the reading passage over counting."""
    from ml.audio import extract_prosody_features

    for name in ("reading", "counting"):
        seg = _segment(transcript, name)
        if seg and seg.get("words"):
            return extract_prosody_features(seg["path"], seg["words"])
    logger.warning("no reading or counting segment with words; prosody features will be zero")
    return {"speech_rate": 0.0, "pause_freq": 0.0, "pause_avg_duration": 0.0}


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


def analyse_call(call_id: int, db_path: str = DB_PATH, calls_dir: Path = CALLS_DIR) -> dict:
    """Analyse one finished call. Returns what it produced, for logging."""
    transcript = _load_transcript(call_id, calls_dir)
    patient_id = transcript["patient_id"]
    flags = [f["reason"] for f in transcript.get("flags", []) if f.get("reason")]

    acoustic = _acoustic_features(transcript, calls_dir, call_id)
    prosody = _prosody_features(transcript)

    from ml.features import build_feature_vector
    features = build_feature_vector(call_id=call_id, acoustic=acoustic, prosody=prosody)

    conn = _connect(db_path)
    try:
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

    result = analyse_call(call_id, db_path=args.db, calls_dir=calls_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
