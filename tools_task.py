import asyncio
import logging
import os
from datetime import datetime
from functools import partial
from zoneinfo import ZoneInfo

import gspread
from google.oauth2.service_account import Credentials

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Google API Scopes ─────────────────────────────────────────────────────────
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ── Paths & Names ─────────────────────────────────────────────────────────────
_CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
_SPREADSHEET_NAME = "Student_Sync_DB"
_WORKSHEET_NAME = "Tasks"

_TZ_JAKARTA = ZoneInfo("Asia/Jakarta")


# ── Sync Core (blocking — always called via asyncio.to_thread) ────────────────

def _append_task_sync(payload: dict) -> list:
    """
    Open the 'Tasks' worksheet of 'Student_Sync_DB' and append one row.

    Row layout (must match the worksheet header):
        Tanggal Dicatat | Nama Tugas | Prioritas | Deadline | Status

    Returns the appended row as a list.
    This is a blocking call — never invoke directly from async code.
    """
    now_jakarta = datetime.now(_TZ_JAKARTA).strftime("%Y-%m-%d")

    row = [
        now_jakarta,
        payload.get("nama_tugas") or payload.get("tugas") or "-",
        payload.get("prioritas") or "Sedang",
        payload.get("deadline") or "-",
        "Belum",
    ]

    creds = Credentials.from_service_account_file(_CREDENTIALS_PATH, scopes=_SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open(_SPREADSHEET_NAME)
    worksheet = spreadsheet.worksheet(_WORKSHEET_NAME)

    append_fn = partial(worksheet.append_row, row, value_input_option="USER_ENTERED")
    append_fn()

    return row


# ── Async Wrapper ─────────────────────────────────────────────────────────────

async def append_task_row(payload: dict) -> dict:
    """
    Async entry-point for appending a task record to Google Sheets.

    Expected payload keys (from AI SHEET_TASK output):
        nama_tugas / tugas  (str) : task name
        prioritas           (str) : "Tinggi" | "Sedang" | "Rendah"
        deadline            (str) : "YYYY-MM-DD" or empty string
        mata_kuliah         (str) : subject / course (informational, not stored separately)

    Returns:
        {"success": True,  "row": [...]}          on success
        {"success": False, "error": "<reason>"}   on failure
    """
    try:
        # Offload blocking gspread I/O to a thread — keeps FastAPI event loop free
        row = await asyncio.to_thread(_append_task_sync, payload)

        logger.info("Task row appended: %s", row)
        return {"success": True, "row": row}

    except FileNotFoundError:
        msg = f"credentials.json tidak ditemukan di '{_CREDENTIALS_PATH}'"
        logger.error(msg)
        return {"success": False, "error": msg}

    except gspread.exceptions.SpreadsheetNotFound:
        msg = f"Spreadsheet '{_SPREADSHEET_NAME}' tidak ditemukan. Periksa nama atau izin sharing."
        logger.error(msg)
        return {"success": False, "error": msg}

    except gspread.exceptions.WorksheetNotFound:
        msg = f"Worksheet '{_WORKSHEET_NAME}' tidak ditemukan di dalam '{_SPREADSHEET_NAME}'."
        logger.error(msg)
        return {"success": False, "error": msg}

    except gspread.exceptions.APIError as exc:
        msg = f"Google Sheets API error: {exc}"
        logger.error(msg)
        return {"success": False, "error": msg}

    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error saat menyimpan tugas: {exc}"
        logger.exception(msg)
        return {"success": False, "error": msg}
