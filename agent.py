import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from groq import AsyncGroq, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# ── Client ────────────────────────────────────────────────────────────────────
_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

_MODEL = "qwen/qwen3.8-27b"

_TZ_JAKARTA = ZoneInfo("Asia/Jakarta")


def get_system_prompt() -> str:
    """Return the system prompt with the current Jakarta timestamp injected."""
    current_date = datetime.now(_TZ_JAKARTA).strftime("%Y-%m-%d %H:%M WIB")
    return f"""Kamu adalah Student-Sync, asisten AI produktivitas khusus mahasiswa IT. Tugas utamamu adalah menganalisis pesan pengguna dan merutekannya ke tiga sistem: Keuangan, Tugas Kuliah, atau Jadwal. Waktu saat ini adalah {current_date}.

Kamu memiliki tiga tools utama. Analisis pesan pengguna dan gunakan logika berikut:
1. SHEET_FINANCE: Gunakan jika teks mengandung unsur transaksi atau uang. Klasifikasikan 'Kategori' secara logis menjadi: Konsumsi, Transportasi, Akademik, atau Hiburan. Payload: {{"tanggal": "YYYY-MM-DD", "deskripsi": "string", "kategori": "string", "nominal": integer, "tipe": "Pengeluaran/Pemasukan"}}
2. SHEET_TASK: Gunakan jika teks mengandung instruksi tugas, PR, assignment, atau laporan. Tentukan 'Prioritas': 'Tinggi' jika deadline < 3 hari, sisanya 'Sedang' atau 'Rendah'. Payload: {{"mata_kuliah": "string", "tugas": "string", "deadline": "YYYY-MM-DD", "prioritas": "string", "status": "Belum Selesai"}}
3. CALENDAR_EVENT: Gunakan jika teks mengandung pertemuan, kelas, rapat, atau jadwal di jam tertentu. Hitung durasi event default selama 1 jam jika tidak disebutkan. Konversikan semua waktu ke format standar ISO 8601. Payload: {{"summary": "string", "start_time": "ISO 8601", "end_time": "ISO 8601"}}

Kamu harus bisa menjalankan beberapa tools sekaligus jika pengguna menggabungkan instruksi dalam satu kalimat.

JANGAN berikan teks pengantar, basa-basi, atau markdown. Kembalikan jawabanmu HANYA dalam format JSON murni dengan struktur ini:
{{
  "actions": [
    {{
      "tool": "[nama tool]",
      "payload": {{ ... }}
    }}
  ],
  "response_text": "[pesan singkat]"
}}"""


async def analyze_intent_with_groq(user_text: str) -> dict:
    """
    Send *user_text* to Groq (LLaMA3-70b) and return a parsed intent dict.

    The system prompt is generated dynamically per-call so the injected
    timestamp always reflects the current Jakarta time.

    Returns a dict with keys 'actions' + 'response_text' on success,
    or 'error' on failure.
    """
    try:
        response = await _client.chat.completions.create(
            model=_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_text},
            ],
            temperature=0.2,      # low temperature → deterministic routing
            max_tokens=1024,      # increased — multi-action responses can be longer
        )

        raw_text = response.choices[0].message.content
        return json.loads(raw_text)

    except json.JSONDecodeError as exc:
        return {
            "error": "JSON_PARSE_ERROR",
            "detail": str(exc),
            "raw": raw_text if "raw_text" in dir() else None,
        }
    except RateLimitError as exc:
        return {"error": "RATE_LIMIT_ERROR", "detail": str(exc)}
    except APIConnectionError as exc:
        return {"error": "CONNECTION_ERROR", "detail": str(exc)}
    except APIError as exc:
        return {"error": "GROQ_API_ERROR", "detail": str(exc), "status_code": exc.status_code}
    except Exception as exc:  # noqa: BLE001 — catch-all safety net
        return {"error": "UNEXPECTED_ERROR", "detail": str(exc)}
