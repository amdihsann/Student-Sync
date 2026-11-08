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


# ── Read ──────────────────────────────────────────────────────────────────────

def _clean_nominal(raw: object) -> float:
    """
    Convert a Nominal value from Google Sheets to float.
    Handles both already-numeric values and formatted strings like 'Rp 25.000'.
    """
    val_str = str(raw).replace("Rp", "").replace(".", "").replace(",", "").replace(" ", "").strip()
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def _build_summary(records: list[dict]) -> dict:
    """
    Pure-Python calculation of finance summary from a list of row dicts.

    Expects each row to have:
        Tanggal (str YYYY-MM-DD) — formatted string, NOT a serial number
        Nominal (str or numeric) — cleaned via _clean_nominal()
        Tipe    (str)            — "Pemasukan" or "Pengeluaran"
    """
    total_masuk:  float = 0.0
    total_keluar: float = 0.0
    # monthly breakdown: {"2026-11": {"Pemasukan": 0.0, "Pengeluaran": 0.0}, ...}
    per_bulan: dict[str, dict[str, float]] = {}

    for row in records:
        nominal = _clean_nominal(row.get("Nominal", 0))
        tipe    = str(row.get("Tipe") or "").strip().lower()
        tanggal = str(row.get("Tanggal") or "")

        # Aggregate totals
        if tipe.startswith("pema"):
            total_masuk += nominal
        elif tipe.startswith("penge"):
            total_keluar += nominal

        # Extract YYYY-MM from formatted date string "YYYY-MM-DD"
        bulan = tanggal[:7] if len(tanggal) >= 7 else "unknown"
        if bulan not in per_bulan:
            per_bulan[bulan] = {"Pemasukan": 0.0, "Pengeluaran": 0.0}
        if tipe.startswith("pema"):
            per_bulan[bulan]["Pemasukan"] += nominal
        elif tipe.startswith("penge"):
            per_bulan[bulan]["Pengeluaran"] += nominal

    return {
        "total_pemasukan":   total_masuk,
        "total_pengeluaran": total_keluar,
        "saldo":             total_masuk - total_keluar,
        "per_bulan":         dict(sorted(per_bulan.items())),  # sorted chronologically
    }


def _get_finance_data_sync() -> tuple[list[dict], dict]:
    """
    Open the 'Finance' worksheet and fetch all rows using default formatting
    so that 'Tanggal' stays as a readable string (e.g. '2026-11-08') instead
    of a Google Sheets serial number.
    Nominal cleaning is handled in Python by _clean_nominal().
    Blocking — always call via asyncio.to_thread().
    """
    creds = Credentials.from_service_account_file(_CREDENTIALS_PATH, scopes=_SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open(_SPREADSHEET_NAME)
    worksheet = spreadsheet.worksheet(_WORKSHEET_NAME)

    records: list[dict] = worksheet.get_all_records()  # default: FORMATTED_VALUE
    summary: dict = _build_summary(records)
    return records, summary


async def get_finance_data() -> dict:
    """
    Async wrapper — fetch all finance records and a pre-computed summary.

    Returns:
        {"success": True,  "data": [...], "summary": {"total_pemasukan": ..., ...}}
        {"success": False, "error": "<reason>"}
    """
    try:
        records, summary = await asyncio.to_thread(_get_finance_data_sync)
        logger.info("Fetched %d finance records. Saldo: %.0f", len(records), summary["saldo"])
        return {"success": True, "data": records, "summary": summary}

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
        msg = f"Unexpected error while fetching finance data: {exc}"
        logger.exception(msg)
        return {"success": False, "error": msg}


# ── Write ─────────────────────────────────────────────────────────────────────

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
