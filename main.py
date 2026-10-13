import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from agent import analyze_intent_with_groq
from tools_finance import append_finance_data
from tools_calendar import create_calendar_event

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_TELEGRAM_API_BASE = f"https://api.telegram.org/bot{_BOT_TOKEN}"

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
        # TODO: implement tools_task.py and call append_task_data(payload) here
        pass

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

async def process_and_reply(chat_id: int | str, user_text: str) -> None:
    """
    Core pipeline (runs in background — never blocks the event loop):
      1. Call Groq AI to analyse intent and get structured actions.
      2. Dispatch each action to the appropriate tool module.
      3. Send the consolidated reply back to the user on Telegram.
    """
    # ── Step 1: AI intent analysis ────────────────────────────────────────────
    ai_result = await analyze_intent_with_groq(user_text)

    # ── Step 2: Handle AI-level errors ───────────────────────────────────────
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

    # ── Step 3: Dispatch each action and collect tool output lines ────────────
    tool_outputs: list[str] = []
    for action in actions:
        output = await _dispatch_action(action)
        if output:  # None means pass/not-yet-implemented
            tool_outputs.append(output)

    # ── Step 4: Build final reply ─────────────────────────────────────────────
    # Lead with the AI's own friendly summary, then append tool confirmations.
    parts: list[str] = []
    if response_text:
        parts.append(response_text)
    if tool_outputs:
        parts.append("\n".join(tool_outputs))

    final_reply = "\n\n".join(parts) if parts else "✅ Selesai diproses."

    await send_telegram_message(chat_id, final_reply)
    logger.info("Reply sent to chat_id=%s | actions=%d", chat_id, len(actions))


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
    text: str = message.get("text", "").strip()

    # Ignore non-text updates (stickers, photos, voice, etc.)
    if not chat_id or not text:
        return JSONResponse(content={"status": "ok"})

    # Delegate to background — endpoint returns immediately
    background_tasks.add_task(process_and_reply, chat_id, text)

    return JSONResponse(content={"status": "ok"})


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    return {"status": "Student-Sync is running 🚀"}
