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
Tugas utamamu adalah menganalisis pesan pengguna dan merutekannya ke satu atau lebih dari enam tools berikut: \
SHEET_FINANCE, SHEET_TASK, CALENDAR_EVENT, READ_FINANCE, READ_TASK, atau READ_CALENDAR. \
Waktu saat ini adalah {current_date}.

============================
ATURAN PER TOOL
============================

── TOOLS TULIS (mencatat data baru) ──

1. SHEET_FINANCE
   Gunakan jika pengguna ingin MENCATAT transaksi, pengeluaran, pemasukan, atau uang.
   - Klasifikasikan field "kategori" menjadi salah satu: Konsumsi, Transportasi, Akademik, atau Hiburan.
   - Jika nominal tidak disebutkan, isi dengan 0.
   - Payload wajib:
     {{"tanggal": "YYYY-MM-DD", "deskripsi": "string", "kategori": "string", "nominal": integer, "tipe": "Pengeluaran atau Pemasukan"}}

2. SHEET_TASK
   Gunakan jika pengguna ingin MENCATAT tugas, PR, assignment, laporan, atau kerjaan kuliah.
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
   Gunakan jika pengguna ingin MENJADWALKAN pertemuan, kelas, rapat, atau event dengan jam tertentu.
   - Konversikan semua waktu ke format ISO 8601 (contoh: "2024-07-15T09:00:00").
   - Jika durasi tidak disebutkan, asumsikan 1 jam (end_time = start_time + 1 jam).
   - Payload wajib:
     {{"summary": "string", "start_time": "ISO 8601", "end_time": "ISO 8601"}}

── TOOLS BACA (mengambil data yang sudah ada) ──

4. READ_FINANCE
   Gunakan jika pengguna BERTANYA tentang riwayat keuangan, total pengeluaran/pemasukan, saldo,
   atau meminta laporan keuangan — bukan mencatat transaksi baru.
   Contoh trigger: "berapa pengeluaranku?", "tampilkan riwayat keuangan", "saldo aku berapa?".
   - Jika kamu memilih tool ini, kosongkan saja field payload. Biarkan sistem mengambilkan datanya untukmu.
   - Payload: {{}}

5. READ_TASK
   Gunakan jika pengguna BERTANYA tentang daftar tugas, tugas yang belum selesai, atau deadline terdekat —
   bukan mencatat tugas baru.
   Contoh trigger: "tugas apa yang belum selesai?", "deadline terdekat apa?", "tampilkan daftar tugasku".
   - Jika kamu memilih tool ini, kosongkan saja field payload. Biarkan sistem mengambilkan datanya untukmu.
   - Payload: {{}}

6. READ_CALENDAR
   Gunakan jika pengguna BERTANYA tentang jadwal, pertemuan, atau kelas — bukan membuat event baru.
   Contoh trigger: "apa jadwalku hari ini?", "besok ada rapat apa?", "jadwal minggu ini".

   Ada DUA jenis payload tergantung pertanyaan user:

   a) Hari SPESIFIK (hari ini, besok, lusa, tanggal tertentu):
      - Konversikan ke YYYY-MM-DD menggunakan {current_date[:10]} sebagai acuan:
          * "hari ini"  → {current_date[:10]}
          * "besok"     → hitung +1 hari dari {current_date[:10]}
          * "lusa"      → hitung +2 hari dari {current_date[:10]}
      - Payload: {{"tanggal": "YYYY-MM-DD", "rentang": ""}}

   b) Rentang TIDAK JELAS ("waktu dekat", "ke depannya", "minggu ini", jadwal tanpa tanggal pasti):
      - Jangan isi tanggal. Isi key "rentang" dengan nilai "upcoming".
      - Payload: {{"tanggal": "", "rentang": "upcoming"}}

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


async def synthesize_answer(user_question: str, raw_data: list | dict) -> str:
    """
    Second-pass LLM call: given the pre-processed data fetched from Google Sheets/Calendar,
    produce a natural-language answer to the user's original question.

    raw_data may be:
    - A list of dicts (e.g. task or calendar event records)
    - A dict containing {"summary": {...}, "records": [...]} for finance queries,
      where 'summary' holds Python-computed totals the LLM must use directly.

    Unlike analyze_intent_with_groq(), this call:
    - Does NOT use response_format=json_object (plain text output)
    - Returns the answer string directly, or an error string on failure
    """
    # Inject current Jakarta time so LLM never claims ignorance of the current date/time
    current_time = datetime.now(_TZ_JAKARTA).strftime("%A, %Y-%m-%d %H:%M WIB")

    prompt = (
        f"Berikut adalah data mentah dan RINGKASAN PERHITUNGAN otomatis dari database:\n"
        f"{raw_data}\n\n"
        f"Jawab pertanyaan user berdasarkan data ini.\n"
        f"PENTING: Gunakan angka dari bagian 'summary' untuk menjawab total pengeluaran/pemasukan. "
        f"JANGAN mencoba menghitung/menjumlahkan sendiri dari raw data untuk menghindari kesalahan.\n\n"
        f"Pertanyaan user:\n{user_question}"
    )
    try:
        response = await _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Kamu adalah Student-Sync, asisten AI mahasiswa. "
                        f"Sebagai AI, kamu wajib tahu bahwa waktu saat ini adalah {current_time}. "
                        f"Jangan pernah katakan kamu tidak tahu waktu atau tanggal hari ini. "
                        f"Jawab pertanyaan user berdasarkan data yang diberikan. "
                        f"Jika data mengandung field 'summary', WAJIB gunakan angka di sana "
                        f"sebagai sumber kebenaran untuk total dan saldo — jangan hitung ulang dari record mentah. "
                        f"Gunakan Bahasa Indonesia yang santai dan informatif. "
                        f"Jangan gunakan markdown. Langsung jawab tanpa basa-basi."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        return "⚠️ Maaf, terlalu banyak permintaan ke AI. Coba lagi sebentar."
    except APIConnectionError:
        return "⚠️ Koneksi ke AI gagal. Periksa koneksi internetmu."
    except APIError as exc:
        return f"⚠️ AI error ({exc.status_code}): {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Terjadi kesalahan tak terduga: {exc}"


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
