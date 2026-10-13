import asyncio
import logging
import os
from datetime import datetime
from functools import partial

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
_WORKSHEET_NAME = "Finance"

# ── Column Order (must match the worksheet header row) ───────────────────────
# Header row in the sheet should be:
#   Tanggal | Deskripsi | Kategori | Nominal | Tipe
_COLUMN_ORDER = ["tanggal", "deskripsi", "kategori", "nominal", "tipe"]


# ── Auth Helper ───────────────────────────────────────────────────────────────

def _get_worksheet() -> gspread.Worksheet:
    """
    Authenticate with a service account and return the target worksheet.
    This is a blocking call — always run via asyncio.to_thread().
    """
    creds = Credentials.from_service_account_file(_CREDENTIALS_PATH, scopes=_SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open(_SPREADSHEET_NAME)
    return spreadsheet.worksheet(_WORKSHEET_NAME)


# ── Core Tool ─────────────────────────────────────────────────────────────────

async def append_finance_data(payload: dict) -> dict:
    """
    Append a finance record to the 'Finance' worksheet of 'Student_Sync_DB'.

    Expected payload keys (all optional — defaults applied if missing):
        tanggal   (str)  : transaction date, e.g. "2024-07-15"
        deskripsi (str)  : short description of the transaction
        kategori  (str)  : category, e.g. "Food", "Transport", "Books"
        nominal   (int/float): amount in IDR
        tipe      (str)  : "Pemasukan" or "Pengeluaran"

    Returns:
        {"success": True, "row": [...]} on success
        {"success": False, "error": "<reason>"} on failure
    """
    # ── Build the row, applying sensible defaults ─────────────────────────────
    today = datetime.now().strftime("%Y-%m-%d")
    row = [
        payload.get("tanggal") or today,
        payload.get("deskripsi") or "-",
        payload.get("kategori") or "Lainnya",
        payload.get("nominal") or 0,
        payload.get("tipe") or "Pengeluaran",
    ]

    try:
        # gspread is synchronous — offload to a thread to keep the event loop free
        worksheet = await asyncio.to_thread(_get_worksheet)

        append_fn = partial(worksheet.append_row, row, value_input_option="USER_ENTERED")
        await asyncio.to_thread(append_fn)

        logger.info("Finance row appended: %s", row)
        return {"success": True, "row": row}

    except FileNotFoundError:
        msg = f"credentials.json not found at '{_CREDENTIALS_PATH}'"
        logger.error(msg)
        return {"success": False, "error": msg}

    except gspread.exceptions.SpreadsheetNotFound:
        msg = f"Spreadsheet '{_SPREADSHEET_NAME}' not found. Check the name or sharing permissions."
        logger.error(msg)
        return {"success": False, "error": msg}

    except gspread.exceptions.WorksheetNotFound:
        msg = f"Worksheet '{_WORKSHEET_NAME}' not found inside '{_SPREADSHEET_NAME}'."
        logger.error(msg)
        return {"success": False, "error": msg}

    except gspread.exceptions.APIError as exc:
        msg = f"Google Sheets API error: {exc}"
        logger.error(msg)
        return {"success": False, "error": msg}

    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error while appending finance data: {exc}"
        logger.exception(msg)
        return {"success": False, "error": msg}
