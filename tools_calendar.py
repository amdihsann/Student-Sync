import asyncio
import logging
import os
from datetime import datetime, timedelta
from functools import partial
from zoneinfo import ZoneInfo

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


# ── Read ──────────────────────────────────────────────────────────────────────

def _get_calendar_data_sync(payload: dict) -> list[dict]:
    """
    List events on the personal calendar based on payload.

    Supports two modes:
      - Single day : payload = {"tanggal": "YYYY-MM-DD", "rentang": ""}
                     Fetches events for that specific date only.
      - Upcoming   : payload = {"tanggal": "", "rentang": "upcoming"}
                     Fetches events from today through the next 7 days.
      - Default    : empty/missing payload → same as today (single day)

    Blocking — always call via asyncio.to_thread().
    """
    if not _CALENDAR_ID:
        raise ValueError("CALENDAR_ID environment variable is not set.")

    tz_jakarta = ZoneInfo("Asia/Jakarta")
    today = datetime.now(tz_jakarta).replace(hour=0, minute=0, second=0, microsecond=0)

    rentang: str = payload.get("rentang", "") or ""
    tanggal: str = payload.get("tanggal", "") or ""

    if rentang.lower() == "upcoming":
        # timeMin = today 00:00, timeMax = today+7 23:59:59
        time_min = today.replace(hour=0,  minute=0,  second=0,  microsecond=0).isoformat()
        time_max = (today + timedelta(days=7)).replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
    else:
        # Single day: use tanggal if provided, otherwise default to today
        if tanggal:
            target_date = datetime.strptime(tanggal, "%Y-%m-%d").replace(tzinfo=tz_jakarta)
        else:
            target_date = today
        time_min = target_date.replace(hour=0,  minute=0,  second=0,  microsecond=0).isoformat()
        time_max = target_date.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()

    creds = Credentials.from_service_account_file(_CREDENTIALS_PATH, scopes=_SCOPES)
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)

    response = (
        service.events()
        .list(
            calendarId=_CALENDAR_ID,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events: list[dict] = []
    for item in response.get("items", []):
        start = item.get("start", {})
        end   = item.get("end", {})
        events.append({
            "summary": item.get("summary", "(Tanpa Judul)"),
            "start":   start.get("dateTime") or start.get("date", ""),
            "end":     end.get("dateTime")   or end.get("date", ""),
        })

    return events


async def get_calendar_data(payload: dict) -> dict:
    """
    Async wrapper — fetch calendar events based on payload.

    Expected payload keys:
        tanggal (str) : "YYYY-MM-DD" for a specific day (empty = today)
        rentang (str) : "upcoming" to fetch events for the next 7 days

    Returns:
        {"success": True,  "data": [...], "mode": "single"|"upcoming"}
        {"success": False, "error": "<reason>"}
    """
    try:
        events = await asyncio.to_thread(_get_calendar_data_sync, payload)
        mode = "upcoming" if (payload.get("rentang") or "").lower() == "upcoming" else "single"
        logger.info("Fetched %d calendar events (mode=%s)", len(events), mode)
        return {"success": True, "data": events, "mode": mode}

    except ValueError as exc:
        msg = str(exc)
        logger.error(msg)
        return {"success": False, "error": msg}

    except FileNotFoundError:
        msg = f"credentials.json tidak ditemukan di '{_CREDENTIALS_PATH}'"
        logger.error(msg)
        return {"success": False, "error": msg}

    except HttpError as exc:
        msg = f"Google Calendar API error {exc.status_code}: {exc.reason}"
        logger.error(msg)
        return {"success": False, "error": msg}

    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error saat membaca calendar: {exc}"
        logger.exception(msg)
        return {"success": False, "error": msg}


# ── Write ─────────────────────────────────────────────────────────────────────

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
