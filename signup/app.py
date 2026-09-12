"""
Patient signup / onboarding — Zone: /signup (owned end-to-end by this workstream).

A caregiver registers a patient for periodic voice check-in calls. Saves the
signup to SQLite (patients + a new patient_signup_details table — see
signup_schema.sql) and prints what would happen next. Does NOT touch
db/schema.sql, db/contracts.py, or db/moss_client.py, and does NOT trigger a
real LiveKit call — call-scheduling isn't built yet.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

DB_PATH = REPO_ROOT / "data" / "app.db"
DB_SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"
SIGNUP_SCHEMA_PATH = BASE_DIR / "signup_schema.sql"

CALL_FREQUENCIES = {"daily", "twice_weekly", "weekly"}

PHONE_RE = re.compile(r"^[0-9()+\-.\s]{7,20}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

app = Flask(__name__, static_folder="static", template_folder="templates")


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the shared `patients` table (from db/schema.sql, read-only) and
    this zone's own patient_signup_details table (signup_schema.sql). Both
    are CREATE TABLE IF NOT EXISTS, so this is safe to run every startup and
    never modifies the shared schema file itself."""
    conn = get_db()
    try:
        conn.executescript(DB_SCHEMA_PATH.read_text())
        conn.executescript(SIGNUP_SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        conn.close()


# Run at import time (not just under `if __name__ == "__main__"`), so the
# table exists whenever this app is imported or served — by a test client,
# by `flask run`, or by a real WSGI server (gunicorn etc. never execute the
# __main__ block). Safe to run every time: init_db() is CREATE TABLE IF NOT
# EXISTS only.
init_db()


def _clean(value) -> str:
    return (value or "").strip()


def validate_signup(data: dict) -> list[str]:
    errors = []

    patient_name = _clean(data.get("patient_name"))
    if not patient_name:
        errors.append("Patient name is required.")

    phone_number = _clean(data.get("phone_number"))
    if not phone_number:
        errors.append("Patient phone number is required.")
    elif not PHONE_RE.match(phone_number):
        errors.append("Patient phone number looks invalid.")

    tz = _clean(data.get("timezone"))
    if not tz:
        errors.append("Timezone is required.")

    preferred_call_times = data.get("preferred_call_times") or []
    if not isinstance(preferred_call_times, list) or not preferred_call_times:
        errors.append("At least one preferred call time is required.")
    else:
        for t in preferred_call_times:
            if not isinstance(t, str) or not TIME_RE.match(t):
                errors.append(f"Invalid call time: {t!r} (expected HH:MM).")
                break

    call_frequency = _clean(data.get("call_frequency")) or "daily"
    if call_frequency not in CALL_FREQUENCIES:
        errors.append(f"Call frequency must be one of {sorted(CALL_FREQUENCIES)}.")

    caregiver_name = _clean(data.get("caregiver_name"))
    if not caregiver_name:
        errors.append("Caregiver name is required.")

    caregiver_email = _clean(data.get("caregiver_email"))
    if not caregiver_email:
        errors.append("Caregiver email is required.")
    elif not EMAIL_RE.match(caregiver_email):
        errors.append("Caregiver email looks invalid.")

    caregiver_phone = _clean(data.get("caregiver_phone"))
    if not caregiver_phone:
        errors.append("Caregiver phone number is required.")
    elif not PHONE_RE.match(caregiver_phone):
        errors.append("Caregiver phone number looks invalid.")

    return errors


@app.route("/")
def index():
    return send_from_directory(BASE_DIR / "templates", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR / "static", filename)


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    errors = validate_signup(data)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    patient_name = _clean(data["patient_name"])
    phone_number = _clean(data["phone_number"])
    tz = _clean(data["timezone"])
    preferred_call_times = data["preferred_call_times"]
    call_frequency = _clean(data.get("call_frequency")) or "daily"
    caregiver_name = _clean(data["caregiver_name"])
    caregiver_email = _clean(data["caregiver_email"])
    caregiver_phone = _clean(data["caregiver_phone"])

    conn = get_db()
    try:
        cur = conn.execute("INSERT INTO patients (name) VALUES (?)", (patient_name,))
        patient_id = cur.lastrowid

        conn.execute(
            """
            INSERT INTO patient_signup_details (
                patient_id, phone_number, timezone, preferred_call_times,
                call_frequency, caregiver_name, caregiver_email, caregiver_phone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                phone_number,
                tz,
                json.dumps(preferred_call_times),
                call_frequency,
                caregiver_name,
                caregiver_email,
                caregiver_phone,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    now = datetime.now(timezone.utc).isoformat()
    times_str = ", ".join(preferred_call_times)
    print(
        f"[{now}] SIGNUP: patient_id={patient_id} '{patient_name}' - "
        f"call would be scheduled at {times_str} ({tz}), frequency={call_frequency}. "
        f"Alerts go to caregiver '{caregiver_name}' <{caregiver_email}>, {caregiver_phone}.",
        flush=True,
    )

    return jsonify(
        {
            "ok": True,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "preferred_call_times": preferred_call_times,
            "timezone": tz,
            "call_frequency": call_frequency,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
