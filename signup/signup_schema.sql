-- Patient signup / onboarding — additive extension owned by /signup.
-- Does NOT modify db/schema.sql, db/contracts.py, or db/moss_client.py.
-- Links to the existing `patients` table (id, name, baseline_call_id,
-- created_at) via patient_id. Safe to run alongside db/schema.sql — this
-- file only ever creates its own table.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patient_signup_details (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id              INTEGER NOT NULL REFERENCES patients(id),
    phone_number            TEXT NOT NULL,
    timezone                TEXT NOT NULL,
    preferred_call_times    TEXT NOT NULL,  -- JSON array of "HH:MM" strings
    call_frequency          TEXT NOT NULL DEFAULT 'daily',
    caregiver_name          TEXT NOT NULL,
    caregiver_email         TEXT NOT NULL,
    caregiver_phone         TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_signup_patient_id ON patient_signup_details(patient_id);
