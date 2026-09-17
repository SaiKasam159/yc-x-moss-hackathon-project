# Parkinson's Voice Check-in Agent (YC x Moss Hackathon)

A voice agent that calls people with Parkinson's, runs a short check-in
script, records their voice, and uses **Moss** to retrieve their past answers
*during* the call — so when something is flagged it can ask a follow-up that
refers to what they said before.

See [CLAUDE.md](CLAUDE.md) for ownership zones before editing.

## What happens on a call

```
browser (/call) or scheduled SIP dial
        │
        ▼
LiveKit room ──► patient audio ──┬──► Deepgram STT (word timestamps)
                                 └──► patient.wav + per-task segments
        │
        ▼
agent/triggers.py  ── flagged? ──► Moss lookup (that patient's earlier calls)
        │  no                                    │
        ▼                                        ▼
next scripted prompt                   OpenAI follow-up question
        └──────────────┬─────────────────────────┘
                       ▼
               edge-tts ──► spoken into the room
                       │
                       └──► live transcript to the call page

when the call ends (agent/post_call.py, separate process):
  segments ──► ml/audio.py ──► ml/features.py ──► ml/predict.py
                                                      │
                          features + score + flags ───┴──► report.txt,
                          SQLite (features, updrs_predictions), Moss
```

The call script: orientation questions → three words to remember → sustained
"ahh" → read a passage → open questions → count to twenty → recall the three
words.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill it in, see below
```

Train the model (downloads the UCI dataset itself):

```bash
python -m ml.train_model
```

## Running it

Two processes. The agent worker:

```bash
set -a && source .env && set +a && python -m agent.agent start
```

The web app (signup form + in-browser call page):

```bash
set -a && source .env && set +a && python -m signup.app
```

Then open <http://localhost:5050> to sign someone up, or
<http://localhost:5050/call> to talk to the agent.

Optional — who is due for a call right now:

```bash
python -m agent.scheduler --once
```

Re-analyse a finished call (also runs automatically when a call ends):

```bash
python -m agent.post_call --latest
python -m agent.post_call --call-id 13
```

Exercise the conversation logic with no LiveKit and no microphone:

```bash
python -m agent.agent --dry-run
```

## What a call leaves behind

```
calls/<id>/patient.wav          the patient's microphone (what the ML reads)
calls/<id>/agent.wav            what the agent said
calls/<id>/segments/*.wav       phonation / reading / counting
calls/<id>/transcript.json      turns, word timestamps, segments, flags
calls/<id>/report.txt           the report
calls/<id>/analysis.log         after-call analysis output
```

Plus rows in `calls`, `features` and `updrs_predictions` in `data/app.db`,
and every answer in Moss under `patient:<id>`.

## Environment

| Variable | Used by | Notes |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | agent, call page | required for real calls |
| `DEEPGRAM_API_KEY` | agent | speech to text |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | agent | follow-up questions; without it a templated question is used |
| `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY` | `db/moss_client.py` | without both, Moss runs in an in-memory stub mode |
| `TTS_PROVIDER`, `EDGE_TTS_VOICE` | agent | `edge-tts` (default, no key). Kokoro is stubbed, not wired |
| `DB_PATH`, `CALLS_DIR` | everything | default `data/app.db`, `calls/` |
| `PATIENT_ID` | agent | pins a patient; normally the call page chooses |
| `LIVEKIT_SIP_TRUNK_ID` | scheduler | only needed to dial real phone numbers |

Timing knobs (`ANSWER_SETTLE_S`, `NO_ANSWER_TIMEOUT_S`, `PHONATION_MAX_S`,
`FALLBACK_TIMEOUT_S`, …) are listed in `agent/agent.py`.

## Read this before showing anyone the UPDRS number

The model does **not** predict UPDRS for a patient it hasn't heard before.
Measured with different patients in train and test (5-fold grouped CV, 42
patients, UCI telemonitoring data):

| | R² | MAE |
|---|---|---|
| always predict the average | −0.06 | 8.8 points |
| 8 voice features, random forest | −0.21 | 9.4 points |
| 8 voice features, SVR | −0.23 | 9.5 points |

Negative R² means worse than guessing the average. An earlier "test R² 0.35"
came from splitting rows at random: each patient contributes ~140 recordings,
so the same people appeared in train and test and the model scored by
recognising individuals. Patient averages span ~10.4 UPDRS points while one
patient varies by ~2.3, so identity is most of the signal. Adding age or sex
makes it worse.

What *is* real: jitter, shimmer, HNR, speech rate and pauses are measured
directly from the recording, and the flags raised during the call come from
what the patient actually said. The report leads with those and labels the
model estimate experimental. Per-patient trends in the measured features are
the defensible direction; absolute severity scoring is not.

## Rotating credentials

The keys currently in `.env` were pasted into a chat during development, so
treat them as exposed and roll them before this goes anywhere real:

- LiveKit — <https://cloud.livekit.io> → project → Settings → Keys
- Deepgram — <https://console.deepgram.com> → API Keys
- OpenAI — <https://platform.openai.com/api-keys>
- Moss — <https://portal.usemoss.dev> → project credentials

`.env` is gitignored and no key has ever been committed (verified against the
full history), but the chat transcript is not a secret store.

## Known limitations

- The SIP dialling path in the scheduler is written but untested — no trunk
  is configured.
- Kokoro TTS is stubbed; edge-tts is what runs.
- RPDE/DFA/PPE are reimplemented in `ml/nonlinear.py` and follow the
  published definitions, but aren't numerically identical to the UCI columns
  they're compared against (see that module's docstring).
- The delayed-recall check flags below 2 of 3 words; that threshold is a
  judgement call, not a validated cutoff.

## Branching

Per `CLAUDE.md`: work happens on branches, not directly on `main`.
