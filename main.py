import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from agent import analyze_intent_with_groq, synthesize_answer
from tools_finance import append_finance_data, get_finance_data
from tools_calendar import create_calendar_event, get_calendar_data
from tools_task import append_task_row, get_task_data

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_TELEGRAM_API_BASE = f"https://api.telegram.org/bot{_BOT_TOKEN}"

_raw_authorized_id = os.getenv("AUTHORIZED_CHAT_ID", "")
try:
    _AUTHORIZED_CHAT_ID: int | None = int(_raw_authorized_id)
except ValueError:
    _AUTHORIZED_CHAT_ID = None
    logger.warning(
        "AUTHORIZED_CHAT_ID is not set or not a valid integer — "
        "all incoming messages will be rejected."
    )

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Student-Sync", version="1.0.0")


# ── Telegram Helper ───────────────────────────────────────────────────────────

async def send_telegram_message(chat_id: int | str, text: str) -> None:
    """Send *text* back to the Telegram user identified by *chat_id*."""
    url = f"{_TELEGRAM_API_BASE}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()


# ── Tool Dispatcher ───────────────────────────────────────────────────────────

async def _dispatch_action(action: dict) -> str:
    """
    Execute a single action returned by the AI and return a status string
    to be collected into the final reply.

    action = {"tool": "<TOOL_NAME>", "payload": { ... }}
    """
    tool = action.get("tool", "").upper()
    payload = action.get("payload", {})

    if tool == "SHEET_FINANCE":
        result = await append_finance_data(payload)
        if result.get("success"):
            row = result["row"]
            return (
                f"💰 *Keuangan tercatat!*\n"
                f"• Tanggal   : {row[0]}\n"
                f"• Deskripsi : {row[1]}\n"
                f"• Kategori  : {row[2]}\n"
                f"• Nominal   : Rp {row[3]:,}\n"
                f"• Tipe      : {row[4]}"
            )
        return f"⚠️ *Gagal mencatat keuangan:* `{result.get('error', 'unknown error')}`"

    if tool == "SHEET_TASK":
        try:
            result = await append_task_row(payload)
            if result.get("success"):
                row = result["row"]
                return (
                    f"📋 *Tugas dicatat!*\n"
                    f"• Tanggal   : {row[0]}\n"
                    f"• Nama Tugas: {row[1]}\n"
                    f"• Prioritas : {row[2]}\n"
                    f"• Deadline  : {row[3]}\n"
                    f"• Status    : {row[4]}"
                )
            return f"⚠️ *Gagal mencatat tugas:* `{result.get('error', 'unknown error')}`"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled exception in SHEET_TASK dispatch: %s", exc)
            return f"⚠️ *Gagal mencatat tugas karena kesalahan tidak terduga:* `{exc}`"

    if tool == "CALENDAR_EVENT":
        try:
            result = await create_calendar_event(payload)
            if result.get("success"):
                return (
                    f"📅 *Event dijadwalkan!*\n"
                    f"• Judul    : {payload.get('summary', '-')}\n"
                    f"• Mulai    : {payload.get('start_time', '-')}\n"
                    f"• Selesai  : {payload.get('end_time', '-')}\n"
                    f"• Link     : {result.get('html_link') or '_tidak tersedia_'}"
                )
            return f"⚠️ *Gagal membuat event:* `{result.get('error', 'unknown error')}`"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled exception in CALENDAR_EVENT dispatch: %s", exc)
            return f"⚠️ *Gagal membuat event karena kesalahan tidak terduga:* `{exc}`"

    logger.warning("Unknown tool received from AI: %s", tool)
    return f"🤔 *Tool `{tool}` belum dikenali atau belum diimplementasikan.*"


# ── Background Task ───────────────────────────────────────────────────────────

_READ_TOOLS = {"READ_FINANCE", "READ_TASK", "READ_CALENDAR"}
_DATA_FETCHERS = {
    "READ_FINANCE":  get_finance_data,
    "READ_TASK":     get_task_data,
    "READ_CALENDAR": get_calendar_data,
}


async def process_and_reply(chat_id: int | str, user_text: str) -> None:
    """
    2-step reasoning pipeline (runs in background — never blocks the event loop):

    WRITE path (SHEET_FINANCE / SHEET_TASK / CALENDAR_EVENT):
      Pass 1 → AI returns structured JSON → dispatch to tool → send response_text

    READ path (READ_FINANCE / READ_TASK / READ_CALENDAR):
      Pass 1 → AI returns READ intent (with optional payload, e.g. tanggal for READ_CALENDAR)
      Pass 2 → fetch data from Sheets/Calendar → feed to LLM → get natural-language answer
             → send that answer to Telegram
    """
    # ── Pass 1: intent routing ────────────────────────────────────────────────
    ai_result = await analyze_intent_with_groq(user_text)

    if "error" in ai_result:
        error_msg = (
            f"⚠️ *Terjadi kesalahan saat memproses pesanmu.*\n"
            f"`{ai_result['error']}: {ai_result.get('detail', '-')}`"
        )
        await send_telegram_message(chat_id, error_msg)
        logger.error("AI error for chat_id=%s: %s", chat_id, ai_result)
        return

    actions: list = ai_result.get("actions", [])
    response_text: str = ai_result.get("response_text", "")

    # ── Separate READ actions from WRITE actions ──────────────────────────────
    read_actions  = [a for a in actions if a.get("tool", "").upper() in _READ_TOOLS]
    write_actions = [a for a in actions if a.get("tool", "").upper() not in _READ_TOOLS]

    # ── WRITE path: dispatch tools, collect confirmations ────────────────────
    write_outputs: list[str] = []
    for action in write_actions:
        output = await _dispatch_action(action)
        if output:
            write_outputs.append(output)

    # ── READ path: fetch data → Pass 2 (LLM synthesizes natural answer) ──────
    read_outputs: list[str] = []
    for action in read_actions:
        tool = action.get("tool", "").upper()
        fetcher = _DATA_FETCHERS.get(tool)
        if not fetcher:
            continue
        try:
            # Pass the action payload — READ_CALENDAR uses {"tanggal": "YYYY-MM-DD"},
            # READ_FINANCE / READ_TASK ignore it (their fetchers accept no arguments).
            action_payload: dict = action.get("payload") or {}
            result = await fetcher(action_payload) if tool == "READ_CALENDAR" else await fetcher()
            if not result.get("success"):
                read_outputs.append(
                    f"⚠️ Gagal mengambil data ({tool}): {result.get('error', 'unknown')}"
                )
                continue
            raw_data: list = result.get("data", [])
            if not raw_data:
                read_outputs.append("📭 Belum ada data yang tersimpan.")
                continue

            # For READ_FINANCE: pass the pre-computed summary + raw records to LLM.
            # The summary (calculated in Python) makes the LLM's job trivial — it
            # only needs to narrate the numbers, not re-calculate them.
            if tool == "READ_FINANCE" and "summary" in result:
                context_data = {
                    "summary": result["summary"],
                    "records": raw_data,
                }
            else:
                context_data = raw_data

            # Pass 2 — synthesize a human-friendly answer from the pre-processed data
            answer = await synthesize_answer(user_text, context_data)
            read_outputs.append(answer)
            logger.info("Second-pass synthesis done for tool=%s chat_id=%s", tool, chat_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in READ path for tool=%s: %s", tool, exc)
            read_outputs.append(f"⚠️ Terjadi kesalahan saat membaca data: {exc}")

    # ── Build final reply ─────────────────────────────────────────────────────
    parts: list[str] = []
    # For WRITE: lead with AI confirmation text, then tool receipts
    if write_outputs:
        if response_text:
            parts.append(response_text)
        parts.extend(write_outputs)
    # For READ: the synthesized answer IS the reply — skip redundant response_text
    parts.extend(read_outputs)

    final_reply = "\n\n".join(parts) if parts else "✅ Selesai diproses."

    await send_telegram_message(chat_id, final_reply)
    logger.info(
        "Reply sent to chat_id=%s | write_actions=%d read_actions=%d",
        chat_id, len(write_actions), len(read_actions),
    )


# ── Webhook Endpoint ──────────────────────────────────────────────────────────

@app.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive a Telegram update, immediately return HTTP 200,
    and delegate all processing to a background task.

    Telegram requires a response within 60 seconds — returning instantly here
    ensures no retry storms even when Groq or Google APIs are slow.
    """
    try:
        body: dict = await request.json()
    except Exception:
        # Malformed JSON — return 200 so Telegram does not retry endlessly
        return JSONResponse(content={"status": "ok"})

    # Support both fresh messages and edited messages
    message = body.get("message") or body.get("edited_message")
    if not message:
        return JSONResponse(content={"status": "ok"})

    chat_id: int = message.get("chat", {}).get("id")

    # ── Security check — silently drop requests from unauthorized users ───────
    if _AUTHORIZED_CHAT_ID is None or chat_id != _AUTHORIZED_CHAT_ID:
        logger.warning("Unauthorized access attempt from chat_id=%s — ignored.", chat_id)
        return JSONResponse(content={"status": "ignored"})

    text: str = message.get("text", "").strip()

    # Ignore non-text updates (stickers, photos, voice, etc.)
    if not text:
        return JSONResponse(content={"status": "ok"})

    # Delegate to background — endpoint returns immediately
    background_tasks.add_task(process_and_reply, chat_id, text)

    return JSONResponse(content={"status": "ok"})


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    return {"status": "Student-Sync is running 🚀"}
