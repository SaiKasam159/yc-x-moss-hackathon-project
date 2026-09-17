"""
Call scheduling — decides who is due for a check-in and starts it.

Signup collects a timezone, preferred call times and a frequency, and then
printed "call would be scheduled at ..." and did nothing with them. Nothing
in the project ever started a call: the only way to have one was for someone
to open the call page and press a button.

This closes that loop as far as it can be closed without a phone account:

    python -m agent.scheduler --once            # who is due right now
    python -m agent.scheduler                   # keep checking every minute
    python -m agent.scheduler --once --place-calls   # actually dial (needs SIP)

Placing real calls is opt-in on purpose. It dials a real phone number and
costs money, so without --place-calls (and a configured LiveKit SIP trunk)
this prints the link that starts the call instead:

    http://localhost:5050/call?patient_id=<id>

To dial for real, set LIVEKIT_SIP_TRUNK_ID to an outbound trunk configured in
your LiveKit Cloud project and pass --place-calls. The agent picks the call
up the same way it does from the browser: the patient id travels in the
participant metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("agent.scheduler")

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("DB_PATH", "data/app.db")
CALL_PAGE = os.environ.get("CALL_PAGE_URL", "http://localhost:5050/call")

# How close to a preferred time counts as "now", and how long since the last
# call each frequency requires.
WINDOW_MINUTES = int(os.environ.get("SCHEDULE_WINDOW_MINUTES", "15"))
MIN_DAYS_BETWEEN = {"daily": 1, "twice_weekly": 3, "weekly": 7}


@dataclass
class DueCall:
    patient_id: int
    name: str
    phone: str
    timezone: str
    local_time: str
    due_at: str
    last_call: Optional[str]


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript((REPO_ROOT / "db" / "schema.sql").read_text())
    return conn


def _hours_since(last_call: Optional[str], now_utc: datetime.datetime) -> Optional[float]:
    if not last_call:
        return None
    try:
        when = datetime.datetime.fromisoformat(last_call.replace("Z", ""))
    except ValueError:
        return None
    return (now_utc - when).total_seconds() / 3600.0


def due_calls(db_path: str = DB_PATH, now_utc: Optional[datetime.datetime] = None) -> list[DueCall]:
    """Patients whose preferred time is within the window and whose frequency
    allows another call."""
    now_utc = now_utc or datetime.datetime.utcnow()
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT p.id, p.name, d.phone_number, d.timezone, d.preferred_call_times,
                   d.call_frequency,
                   (SELECT MAX(c.timestamp) FROM calls c WHERE c.patient_id = p.id)
              FROM patients p
              JOIN patient_signup_details d ON d.patient_id = p.id
            """
        ).fetchall()
    except sqlite3.Error as exc:  # no signups table yet
        logger.warning("could not read signups: %s", exc)
        return []
    finally:
        conn.close()

    due: list[DueCall] = []
    for pid, name, phone, tz_name, times_json, frequency, last_call in rows:
        try:
            tz = ZoneInfo(tz_name) if tz_name else None
        except Exception:
            logger.warning("patient %s has an unknown timezone %r; skipping", pid, tz_name)
            continue
        local_now = now_utc.replace(tzinfo=datetime.timezone.utc).astimezone(tz) if tz else now_utc

        try:
            wanted = json.loads(times_json) or []
        except (TypeError, json.JSONDecodeError):
            wanted = []

        hours = _hours_since(last_call, now_utc)
        needed = MIN_DAYS_BETWEEN.get(frequency or "daily", 1) * 24
        if hours is not None and hours < needed:
            continue  # called recently enough for their frequency

        for slot in wanted:
            try:
                hh, mm = (int(x) for x in slot.split(":"))
            except (ValueError, AttributeError):
                continue
            slot_today = local_now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            minutes_past = (local_now - slot_today).total_seconds() / 60.0
            if 0 <= minutes_past <= WINDOW_MINUTES:
                due.append(DueCall(
                    patient_id=pid, name=name, phone=phone, timezone=tz_name or "server time",
                    local_time=local_now.strftime("%Y-%m-%d %H:%M"), due_at=slot,
                    last_call=last_call,
                ))
                break
    return due


async def place_call(call: DueCall) -> str:
    """Dial the patient over LiveKit SIP. Requires an outbound trunk."""
    trunk = os.environ.get("LIVEKIT_SIP_TRUNK_ID")
    if not trunk:
        raise RuntimeError(
            "LIVEKIT_SIP_TRUNK_ID is not set — configure an outbound SIP trunk in "
            "LiveKit Cloud to dial real phone numbers."
        )
    from livekit import api

    room = f"checkin-{call.patient_id}-{datetime.datetime.utcnow():%Y%m%d-%H%M%S}"
    lk = api.LiveKitAPI()
    try:
        await lk.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk,
                sip_call_to=call.phone,
                room_name=room,
                participant_identity=f"patient-{call.patient_id}",
                participant_name=call.name,
                # Same channel the call page uses, so the agent knows who it called.
                participant_metadata=json.dumps({"patient_id": call.patient_id}),
                wait_until_answered=True,
            )
        )
    finally:
        await lk.aclose()
    return room


async def run_once(db_path: str, place: bool) -> list[DueCall]:
    calls = due_calls(db_path)
    if not calls:
        logger.info("nobody is due right now")
        return []
    for call in calls:
        logger.info(
            "due: %s (#%s) at %s %s — local time %s, last call %s",
            call.name, call.patient_id, call.due_at, call.timezone,
            call.local_time, call.last_call or "never",
        )
        if not place:
            logger.info("   start it here: %s?patient_id=%s", CALL_PAGE, call.patient_id)
            continue
        try:
            room = await place_call(call)
            logger.info("   dialled %s, room %s", call.phone, room)
        except Exception as exc:
            logger.error("   could not dial %s: %s", call.phone, exc)
    return calls


async def main_async(args) -> None:
    if args.once:
        await run_once(args.db, args.place_calls)
        return
    logger.info("watching for due calls every %ss (Ctrl-C to stop)", args.interval)
    while True:
        try:
            await run_once(args.db, args.place_calls)
        except Exception:
            logger.exception("scheduler pass failed")
        await asyncio.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start check-in calls when patients are due.")
    parser.add_argument("--once", action="store_true", help="check once and exit")
    parser.add_argument("--interval", type=int, default=60, help="seconds between checks")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--place-calls", action="store_true",
                        help="actually dial (needs LIVEKIT_SIP_TRUNK_ID); otherwise prints the link")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
