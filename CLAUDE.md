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

[paste the FeatureVector fields, MossQARecord fields, schema.sql column names here]

## Current stub behavior

`/ml/predict.py` currently returns a stubbed fake UPDRS score — this is
intentional until Workstream B replaces it. Do not "fix" this by rewriting
`agent.py`'s expectations; the contract stays fixed, only the stub
implementation changes.

## Branch discipline

Work happens on `voice-agent` or `ml-pipeline` branches, not directly on
`main`. Do not merge branches automatically — flag when a PR is ready.
