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
    return f"""Kamu adalah Student-Sync, asisten AI produktivitas khusus mahasiswa IT. \
Tugas utamamu adalah menganalisis pesan pengguna dan merute­kannya ke satu atau lebih dari tiga sistem berikut: Keuangan, Tugas Kuliah, atau Jadwal. \
Waktu saat ini adalah {current_date}.

============================
ATURAN PER TOOL
============================

1. SHEET_FINANCE
   Gunakan jika teks mengandung unsur transaksi, pengeluaran, pemasukan, atau uang.
   - Klasifikasikan field "kategori" menjadi salah satu: Konsumsi, Transportasi, Akademik, atau Hiburan.
   - Jika nominal tidak disebutkan, isi dengan 0.
   - Payload wajib:
     {{"tanggal": "YYYY-MM-DD", "deskripsi": "string", "kategori": "string", "nominal": integer, "tipe": "Pengeluaran atau Pemasukan"}}

2. SHEET_TASK
   Gunakan jika teks mengandung instruksi tugas, PR, assignment, laporan, atau kerjaan kuliah.
   - Ekstrak "nama_tugas" sebagai nama spesifik tugas yang dimaksud.
   - Ekstrak "mata_kuliah" dari konteks; isi string kosong jika tidak disebutkan.
   - Konversi deadline ke format YYYY-MM-DD. Jika tidak disebutkan, isi string kosong.
   - Tentukan "prioritas" berdasarkan selisih hari antara {current_date[:10]} dan deadline:
       * Kurang dari 3 hari  → "Tinggi"
       * 3 sampai 7 hari     → "Sedang"
       * Lebih dari 7 hari   → "Rendah"
       * Deadline tidak ada  → "Sedang" (default)
   - Payload wajib:
     {{"nama_tugas": "string", "mata_kuliah": "string", "deadline": "YYYY-MM-DD atau string kosong", "prioritas": "Tinggi/Sedang/Rendah"}}

3. CALENDAR_EVENT
   Gunakan jika teks mengandung pertemuan, kelas, rapat, atau jadwal dengan jam tertentu.
   - Konversikan semua waktu ke format ISO 8601 (contoh: "2024-07-15T09:00:00").
   - Jika durasi tidak disebutkan, asumsikan 1 jam (end_time = start_time + 1 jam).
   - Payload wajib:
     {{"summary": "string", "start_time": "ISO 8601", "end_time": "ISO 8601"}}

============================
ATURAN OUTPUT
============================

- Kamu HARUS bisa mengaktifkan beberapa tools sekaligus jika pengguna menggabungkan instruksi dalam satu pesan.
- JANGAN berikan teks pengantar, basa-basi, penjelasan, atau markdown fence.
- Kembalikan jawabanmu HANYA dalam format JSON murni berikut — tidak ada karakter lain di luar JSON:

{{
  "actions": [
    {{
      "tool": "NAMA_TOOL",
      "payload": {{ ... }}
    }}
  ],
  "response_text": "Pesan konfirmasi singkat dalam Bahasa Indonesia untuk dikirim ke pengguna."
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
