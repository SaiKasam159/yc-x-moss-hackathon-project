# Parkinson's Voice Check-in Agent (YC x Moss Hackathon)

An AI voice agent that places periodic phone check-in calls to Parkinson's
patients, runs a fixed script of recall/phonation/reading/Q&A/counting tasks,
extracts acoustic biomarkers from the audio, predicts UPDRS progression, and
uses **Moss** for real-time semantic retrieval of the patient's past answers
*during* the live call — so the agent can ask an informed follow-up question
the moment it detects an inconsistent answer or a flagged symptom.

See [CLAUDE.md](CLAUDE.md) for ownership zones and branch discipline before
editing anything.

## Architecture

```
LiveKit room
   │
   ▼
Deepgram STT (word-level timestamps)
   │
   ▼
agent/triggers.py  ──fired?──▶  db/moss_client.py (scoped Moss query)
   │  not fired                        │
   ▼                                   ▼
agent/call_script.py            LLM (informed follow-up)
   (next scripted prompt)              │
   └───────────────┬────────────────────┘
                    ▼
           TTS (Kokoro-82M, fallback edge-tts)
                    ▼
              back into the room

After the call:
  ml/audio.py → ml/features.py → ml/predict.py (UPDRS) → reports/generate_report.py
  (all persisted via db/schema.sql, report also pushed back into Moss)
```

## Layout

- `/db` — shared contracts (schema, dataclasses, Moss client). **Fully implemented** and runnable now (Moss client runs in an in-memory stub mode until real keys are set).
- `/agent` — Zone A: LiveKit call loop, fixed call script, trigger logic. Skeleton/stubs — no real LiveKit session wiring yet.
- `/ml` — Zone B: acoustic feature extraction, UPDRS model training/prediction. Skeleton/stubs — `predict.py` returns a fake score on purpose until Workstream B trains a real model.
- `/reports` — LLM-generated call reports, pushed to Moss. Skeleton/stub.
- `/data` — UCI Parkinson's Telemonitoring dataset goes here (not downloaded yet, gitignored).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then fill in `.env` — see **Required API keys** below.

### Sanity check (no keys needed)

```bash
python3 -m agent.agent          # dry-runs handle_turn() against the stub Moss store
python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); c.executescript(open('db/schema.sql').read()); print('schema OK')"
```

## Required API keys

| Key(s) | Used in | Notes |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | `agent/agent.py` | LiveKit Cloud project — needed to run a real call session |
| `DEEPGRAM_API_KEY` | `agent/agent.py` | STT, word-level timestamps enabled for pause detection |
| `MOSS_API_KEY`, `MOSS_ENDPOINT` | `db/moss_client.py` | Without these, runs in an in-memory stub mode so everything else stays testable |
| `LLM_API_KEY` (provider TBD) | `agent/agent.py`, `reports/generate_report.py` | **Open question** — which LLM provider? Not yet decided; add the right env var once chosen |
| *(none)* `TTS_PROVIDER` | `agent/agent.py` | `kokoro` (self-hosted, no key) or `edge-tts` (no key) — fallback if Kokoro setup is slow to get working |

Nothing above has been installed or requested yet — flagging per your ask, so you can supply keys / confirm the LLM provider before those files go further.

## Open questions for you

1. **LLM provider** for follow-up generation and report writing — not specified in the stack. Anthropic/OpenAI/other?
2. **Kokoro-82M self-hosting** — local process, or a small server? Affects `KOKORO_MODEL_PATH` vs `KOKORO_SERVER_URL` in `.env.example`.
3. **UPDRS target column** — UCI dataset has both `motor_UPDRS` and `total_UPDRS`; `ml/train_model.py` currently defaults to `total_UPDRS`, confirm that's right.

## Branching

Per `CLAUDE.md`: work happens on `voice-agent` (Zone A) and `ml-pipeline` (Zone B) branches, not directly on `main`.
