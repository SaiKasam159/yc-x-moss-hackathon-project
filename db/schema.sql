-- Parkinson's Voice Check-in Agent — SQLite schema
-- Shared contract between Zone A (voice agent) and Zone B (ML pipeline).
-- Do not change column names/types without confirming with both workstream owners.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    baseline_call_id    INTEGER REFERENCES calls(id),
    created_at          TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL REFERENCES patients(id),
    timestamp           TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    audio_path          TEXT,
    transcript_path     TEXT
);

CREATE TABLE IF NOT EXISTS features (
    call_id             INTEGER PRIMARY KEY REFERENCES calls(id),
    jitter_local        REAL,
    jitter_rap          REAL,
    shimmer_local       REAL,
    shimmer_apq5        REAL,
    hnr                 REAL,
    rpde                REAL,
    dfa                 REAL,
    ppe                 REAL,
    speech_rate         REAL,
    pause_freq          REAL,
    pause_avg_duration  REAL
);

CREATE TABLE IF NOT EXISTS updrs_predictions (
    call_id             INTEGER PRIMARY KEY REFERENCES calls(id),
    predicted_score     REAL NOT NULL,
    confidence_band     TEXT,
    anomaly_flag        INTEGER NOT NULL DEFAULT 0  -- 0/1 boolean
);

CREATE INDEX IF NOT EXISTS idx_calls_patient_id ON calls(patient_id);
