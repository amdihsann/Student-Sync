import asyncio
import logging
import os
from functools import partial

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Google API Scopes ─────────────────────────────────────────────────────────
_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# ── Paths & Config ────────────────────────────────────────────────────────────
_CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")

# The personal Google Calendar that has been shared with the service account.
# MUST be set in .env — using 'primary' would target the service account's own calendar.
_CALENDAR_ID = os.getenv("CALENDAR_ID", "")

_TIMEZONE = "Asia/Jakarta"


# ── Blocking helper (runs inside asyncio.to_thread) ───────────────────────────

def _insert_event(event_body: dict) -> dict:
    """
    Build the Calendar service and insert the event.
    This is fully synchronous (blocking I/O) — must never be called directly
    from an async context.  Always invoke via asyncio.to_thread().
    """
    if not _CALENDAR_ID:
        raise ValueError("CALENDAR_ID environment variable is not set.")

    creds = Credentials.from_service_account_file(_CREDENTIALS_PATH, scopes=_SCOPES)
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)

    insert_fn = partial(
        service.events().insert,
        calendarId=_CALENDAR_ID,
        body=event_body,
    )
    # .execute() is also blocking — wrap both build+execute in one thread call
    return insert_fn().execute()


# ── Core Tool ─────────────────────────────────────────────────────────────────

async def create_calendar_event(payload: dict) -> dict:
    """
    Create a new event on Google Calendar from the AI-produced payload.

    Expected payload keys:
        summary    (str) : event title
        start_time (str) : ISO 8601 datetime, e.g. "2024-07-15T09:00:00"
        end_time   (str) : ISO 8601 datetime, e.g. "2024-07-15T10:00:00"

    Returns:
        {"success": True,  "event_id": "<id>", "html_link": "<url>"} on success
        {"success": False, "error": "<reason>"}                       on failure
    """
    summary: str = payload.get("summary") or "Event (Student-Sync)"
    start_time: str = payload.get("start_time", "")
    end_time: str = payload.get("end_time", "")

    if not start_time or not end_time:
        msg = "Payload harus menyertakan 'start_time' dan 'end_time' dalam format ISO 8601."
        logger.error(msg)
        return {"success": False, "error": msg}

    # ── Build Calendar API event body ─────────────────────────────────────────
    # timeZone MUST be declared in both 'start' and 'end' blocks per the
    # Calendar API spec — the API ignores a top-level timezone field.
    event_body = {
        "summary": summary,
        "start": {
            "dateTime": start_time,
            "timeZone": _TIMEZONE,
        },
        "end": {
            "dateTime": end_time,
            "timeZone": _TIMEZONE,
        },
    }

    try:
        # Offload blocking SDK call to a thread — keeps FastAPI event loop free
        created_event: dict = await asyncio.to_thread(_insert_event, event_body)

        event_id: str = created_event.get("id", "")
        html_link: str = created_event.get("htmlLink", "")

        logger.info(
            "Calendar event created: id=%s summary='%s' start=%s",
            event_id,
            summary,
            start_time,
        )
        return {"success": True, "event_id": event_id, "html_link": html_link}

    except FileNotFoundError:
        msg = f"credentials.json tidak ditemukan di '{_CREDENTIALS_PATH}'"
        logger.error(msg)
        return {"success": False, "error": msg}

    except HttpError as exc:
        msg = f"Google Calendar API error {exc.status_code}: {exc.reason}"
        logger.error(msg)
        return {"success": False, "error": msg}

    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error saat membuat calendar event: {exc}"
        logger.exception(msg)
        return {"success": False, "error": msg}
