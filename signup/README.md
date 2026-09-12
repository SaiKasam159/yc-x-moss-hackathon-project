# /signup — Patient Onboarding

A caregiver signup form: register a patient for periodic voice check-in
calls and provide caregiver contact info for alerts.

Owns its own zone end-to-end. Does not modify `/db/schema.sql`,
`/db/contracts.py`, or `/db/moss_client.py` — it only reads `patients`
(defined there) and adds one new table via `signup_schema.sql`:
`patient_signup_details`, linked to `patients` by `patient_id`.

## Run it

```bash
pip install -r requirements.txt
python3 -m signup.app
```

Open http://localhost:5050.

Signups are saved to `data/app.db` (gitignored). Call-scheduling isn't
built yet, so a successful signup just logs what would happen next, e.g.:

```
[2026-09-11T...] SIGNUP: patient_id=1 'Jane Doe' — call would be scheduled
at 09:00, 18:00 (America/New_York), frequency=daily. Alerts go to
caregiver 'John Doe' <john@example.com>, (555) 987-6543.
```

No real LiveKit call is triggered.
