# Project: Parkinson's Voice Check-in Agent (YC x Moss Hackathon)

## Ownership boundaries — DO NOT CROSS

This repo is split between two developers working in parallel. Before editing
any file, check which zone it's in:

**Zone A — Voice Agent (owner: [your name])**
`/agent/**` — do not modify unless working on the live call loop, script state
machine, or trigger logic.

**Zone B — ML Pipeline (owner: [friend's name])**
`/ml/**`, `/reports/**` — do not modify unless working on audio processing,
feature extraction, or the UPDRS model.

**Shared — ASK BEFORE EDITING, changes affect both zones**
`/db/schema.sql`, `/db/contracts.py`, `/db/moss_client.py` — these are the fixed
contracts both zones depend on. Do not change field names, types, or function
signatures in these files without explicit confirmation from the user, even
if it seems like a small fix. If a change here seems necessary, stop and ask
first rather than editing.

## Fixed contracts (do not deviate)

`db/schema.sql` tables and columns:
- `patients(id, name, baseline_call_id, created_at)`
- `calls(id, patient_id, timestamp, audio_path, transcript_path)`
- `features(call_id, jitter_local, jitter_rap, shimmer_local, shimmer_apq5, hnr, rpde, dfa, ppe, speech_rate, pause_freq, pause_avg_duration)`
- `updrs_predictions(call_id, predicted_score, confidence_band, anomaly_flag)`

`db/contracts.py` dataclasses (mirror the tables above field-for-field):
- `Patient(name, id, baseline_call_id, created_at)`
- `Call(patient_id, id, timestamp, audio_path, transcript_path)`
- `FeatureVector(call_id, jitter_local, jitter_rap, shimmer_local, shimmer_apq5, hnr, rpde, dfa, ppe, speech_rate, pause_freq, pause_avg_duration)`
- `UPDRSPrediction(call_id, predicted_score, confidence_band, anomaly_flag)`
- `MossQARecord(patient_id, call_id, question_id, question_topic, answer_text, timestamp, extra)` — Moss-only, not persisted to SQLite

`db/moss_client.py` functions (all `async def` — the real Moss SDK is async-only):
- `ingest_qa_record(record: MossQARecord) -> None`
- `query_patient_history(patient_id, query_text, question_topic=None, top_k=5) -> list[MossQARecord]`
- `push_patient_session(patient_id: int) -> None` — persists a patient's session to the cloud at end of call
- Backed by one `moss.SessionIndex` per patient (`f"patient:{patient_id}"`), via `MOSS_PROJECT_ID`/`MOSS_PROJECT_KEY`

## Current stub behavior

`/ml/predict.py` currently returns a stubbed fake UPDRS score — this is
intentional until Workstream B replaces it. Do not "fix" this by rewriting
`agent.py`'s expectations; the contract stays fixed, only the stub
implementation changes.

## Branch discipline

Work happens on `voice-agent` (Zone A) or `development` (Zone B), not
directly on `main`. Do not merge branches automatically — flag when a PR is
ready.
