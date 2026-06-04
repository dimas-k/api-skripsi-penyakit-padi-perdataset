"""
llm.py — LLM dengan Integrasi RAG
===================================
3 Tier Model LLM:
  LOW    : Qwen 2.5-3B        (Ollama, lokal)  — model kecil, sangat cepat
  MEDIUM : Gemini 2.5 Flash   (Google API)     — seimbang, cloud, context 1M token
  HIGH   : Llama 3.3-70B      (Groq API)       — model besar, kualitas tinggi, cepat via cloud

Cara menjalankan Ollama (untuk LOW tier):
  ollama serve
  ollama pull qwen2.5:3b

Variabel .env yang dibutuhkan:
  GEMINI_API_KEY=...
  GROQ_API_KEY=...
  OLLAMA_BASE_URL=http://localhost:11434  (opsional, default lokal)
"""

import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
# KONFIGURASI 3 TIER MODEL LLM
# ═════════════════════════════════════════════════════════════════

# ── Ollama (LOW) ───────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_LOW_MODEL    = "qwen2.5:3b"                 # LOW  — Ollama lokal, cepat

# ── Google Gemini (MEDIUM) ─────────────────────────────────────
_GEMINI_KEY  = os.getenv("GEMINI_API_KEY", "")
if not _GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY belum diset di .env")

_GEMINI_BASE = "https://generativelanguage.googleapis.com"
LLM_MEDIUM_MODEL = "gemini-2.5-flash"

_GEMINI_CANDIDATES = [
    ("gemini-2.5-flash",      "v1beta"),
    ("gemini-2.5-flash",      "v1"),
    ("gemini-2.0-flash",      "v1"),
    ("gemini-2.0-flash",      "v1beta"),
    ("gemini-2.0-flash-lite", "v1"),
]

GEMINI_MODEL_NAME    = _GEMINI_CANDIDATES[0][0]
_gemini_working: tuple[str, str] | None = None

# ── Groq (HIGH) ────────────────────────────────────────────────
_GROQ_KEY        = os.getenv("GROQ_API_KEY", "")
if not _GROQ_KEY:
    raise ValueError("GROQ_API_KEY belum diset di .env")

_GROQ_BASE       = "https://api.groq.com/openai/v1/chat/completions"
LLM_HIGH_MODEL   = "llama-3.3-70b-versatile"   # HIGH — Groq API, akurat & cepat

# Alias backward compat (kode lama yang pakai GROQ_MODEL_NAME)
GROQ_MODEL_NAME  = LLM_HIGH_MODEL

print(f"✅ LLM siap:")
print(f"   LOW    → {LLM_LOW_MODEL} (Ollama lokal)")
print(f"   MEDIUM → {LLM_MEDIUM_MODEL} (Google Gemini API)")
print(f"   HIGH   → {LLM_HIGH_MODEL} (Groq API)")
print(f"   Ollama URL: {OLLAMA_BASE_URL}")


# ═════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Kamu adalah asisten pertanian yang membantu petani padi langsung di lapangan.

Gunakan bahasa yang sederhana, hangat, dan mudah dipahami oleh petani biasa.
Hindari istilah ilmiah yang rumit — kalau terpaksa pakai, jelaskan artinya dengan singkat.
Jawab langsung ke solusi yang bisa dikerjakan hari ini, bukan teori panjang.
Jika tersedia informasi dari basis pengetahuan, gunakan informasi tersebut sebagai acuan utama.

Kelas yang bisa dideteksi sistem (14 kelas):
- bacterial_leaf_blight    : Hawar Daun Bakteri
- bacterial_leaf_streak    : Hawar Daun Bergaris Bakteri
- bacterial_panicle_blight : Hawar Malai Bakteri
- brown_spot               : Bercak Coklat
- dead_heart               : Batang Mati (penggerek batang)
- downy_mildew             : Embun Bulu
- healthy                  : Tanaman Sehat
- hispa                    : Hispa Padi
- leaf_blast               : Blas Daun
- leaf_smut                : Gosong Palsu Daun
- neck_blast               : Blas Leher Malai
- sheath_blight            : Busuk Pelepah
- tungro                   : Tungro
- harvest_stage            : Fase Panen (siap dipanen)

Untuk kelas harvest_stage, fokus pada: konfirmasi kesiapan panen, langkah panen,
penanganan pascapanen (perontokan, pengeringan, penyimpanan), dan risiko kualitas gabah.

Selalu jawab dalam Bahasa Indonesia."""


# ═════════════════════════════════════════════════════════════════
# INTERNAL: Ollama REST call
# ═════════════════════════════════════════════════════════════════
def _ollama_generate(model: str, prompt: str, base_url: str = None) -> str:
    """
    Panggil Ollama local API (POST /api/chat).
    Timeout 600s untuk mengakomodasi model 70B yang lambat.
    """
    url = f"{base_url or OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model"   : model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream" : False,
        "options": {"temperature": 0.7, "num_predict": 2048},
    }
    try:
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Ollama tidak bisa diakses di {url}.\n"
            f"Pastikan Ollama sudah berjalan: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama timeout (>600s) untuk model {model}.\n"
            f"Model mungkin belum di-pull: ollama pull {model}"
        )
    except KeyError:
        raise RuntimeError(f"Respons Ollama tidak valid untuk model {model}")


# ═════════════════════════════════════════════════════════════════
# INTERNAL: Gemini REST call
# ═════════════════════════════════════════════════════════════════
def _gemini_generate(prompt: str) -> tuple[str, str]:
    """
    Panggil Gemini API dengan fallback ke model/versi berikutnya jika gagal.
    Returns: (answer_text, model_name_used)
    """
    global GEMINI_MODEL_NAME, LLM_MEDIUM_MODEL, _gemini_working

    if _gemini_working:
        active_model, active_ver = _gemini_working
        ordered = [(active_model, active_ver)] + [
            (m, v) for m, v in _GEMINI_CANDIDATES
            if not (m == active_model and v == active_ver)
        ]
    else:
        ordered = list(_GEMINI_CANDIDATES)

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    last_err = None
    for model, ver in ordered:
        url = f"{_GEMINI_BASE}/{ver}/models/{model}:generateContent?key={_GEMINI_KEY}"
        try:
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code == 200:
                data  = r.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text  = "".join(p.get("text", "") for p in parts).strip()
                if not text:
                    finish_reason = data.get("candidates", [{}])[0].get("finishReason", "UNKNOWN")
                    logger.warning(f"Gemini {model} [{ver}]: respons kosong, reason={finish_reason}")
                    last_err = f"Respons kosong (finishReason={finish_reason})"
                    continue
                if model != GEMINI_MODEL_NAME or _gemini_working is None:
                    if _gemini_working is not None:
                        print(f"✅ Gemini fallback berhasil → {model} [{ver}]")
                    GEMINI_MODEL_NAME = model
                    LLM_MEDIUM_MODEL  = model
                    _gemini_working   = (model, ver)
                return text, model
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (403, 404, 429, 500, 502, 503, 504):
                print(f"⚠️  Gemini {model} [{ver}]: {r.status_code} → coba model berikutnya")
                continue
            raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
        except requests.exceptions.Timeout:
            last_err = "Request timeout (>60s)"
            continue
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            continue

    raise RuntimeError(
        f"Semua model Gemini gagal.\nError terakhir: {last_err}\n"
        f"Cek API key di: https://aistudio.google.com/apikey"
    )


# ═════════════════════════════════════════════════════════════════
# INTERNAL: Groq REST call (HIGH)
# ═════════════════════════════════════════════════════════════════
def _groq_generate(prompt: str) -> str:
    """
    Panggil Groq API dengan model Llama 3.3-70B.
    Menggunakan OpenAI-compatible endpoint Groq.
    Timeout 60s — Groq sangat cepat bahkan untuk model 70B.
    """
    headers = {
        "Authorization": f"Bearer {_GROQ_KEY}",
        "Content-Type" : "application/json",
    }
    payload = {
        "model"      : LLM_HIGH_MODEL,
        "messages"   : [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens" : 2048,
    }
    try:
        r = requests.post(_GROQ_BASE, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Groq timeout (>60s) untuk model {LLM_HIGH_MODEL}."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Groq HTTP error: {e}\n"
            f"Response: {r.text[:300]}\n"
            f"Cek API key di: https://console.groq.com/keys"
        )
    except (KeyError, IndexError):
        raise RuntimeError(f"Respons Groq tidak valid: {r.text[:300]}")


# ═════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═════════════════════════════════════════════════════════════════
def _build_recommendation_prompt(disease_name: str, sensor_data: dict | None = None) -> str:
    sensor_section = ""
    if sensor_data:
        sensor_section = f"""
Data Sensor Lapangan Saat Ini:
- Suhu Udara       : {sensor_data.get('suhu_udara', 'N/A')} °C
- Kelembaban Udara : {sensor_data.get('kelembaban_udara', 'N/A')} %
- Suhu Tanah       : {sensor_data.get('suhu_tanah', 'N/A')} °C
- Kelembaban Tanah : {sensor_data.get('kelembaban_tanah', 'N/A')} %
- pH Tanah         : {sensor_data.get('ph_tanah', 'N/A')}
- Nitrogen (N)     : {sensor_data.get('nitrogen', 'N/A')} mg/kg
- Fosfor (P)       : {sensor_data.get('fosfor', 'N/A')} mg/kg
- Kalium (K)       : {sensor_data.get('kalium', 'N/A')} mg/kg
- Intensitas Cahaya: {sensor_data.get('intensitas_cahaya', 'N/A')} lux
- Curah Hujan      : {sensor_data.get('curah_hujan', 'N/A')} mm/hari

Gunakan data sensor ini untuk menilai apakah kondisi lapangan mendukung perkembangan penyakit.
"""
    return f"""Kamera AI mendeteksi penyakit pada tanaman padi: **{disease_name}**
{sensor_section}
Tolong bantu saya dengan penjelasan berikut:
1. Apa itu penyakit ini dan kenapa bisa muncul?
2. Tanda-tanda apa yang biasanya terlihat di tanaman?
3. Apa yang harus saya lakukan sekarang untuk mengatasi ini?
4. Bagaimana cara mencegahnya agar tidak muncul lagi?
5. Apakah kondisi sensor di atas membuat penyakit ini makin parah atau tidak? (hanya jika data sensor tersedia)

Jawab dengan bahasa yang mudah dipahami dan langsung bisa diterapkan.
"""


def _build_rag_prompt(disease_name: str, sensor_data: dict | None = None) -> tuple[str, list[dict]]:
    try:
        from rag import build_rag_prompt
        prompt, chunks = build_rag_prompt(disease_name=disease_name, sensor_data=sensor_data)
        return prompt, chunks
    except ImportError:
        logger.warning("Modul rag.py tidak ditemukan, fallback ke prompt tanpa RAG.")
        return _build_recommendation_prompt(disease_name, sensor_data), []
    except Exception as e:
        logger.warning(f"RAG gagal ({e}), fallback ke prompt tanpa RAG.")
        return _build_recommendation_prompt(disease_name, sensor_data), []


# ═════════════════════════════════════════════════════════════════
# FUNGSI REKOMENDASI — 3 TIER MODEL
# ═════════════════════════════════════════════════════════════════

def get_recommendation_low(
    disease_name: str,
    sensor_data : dict | None = None,
) -> tuple[str, list[dict]]:
    """
    LOW tier: Qwen2.5-3B via Ollama (lokal).
    Model kecil (~3B), sangat cepat, cocok untuk edge device.
    Returns: (answer_text, retrieved_chunks)
    """
    prompt, chunks = _build_rag_prompt(disease_name, sensor_data)
    text = _ollama_generate(LLM_LOW_MODEL, prompt)
    return text, chunks


def get_recommendation_medium(
    disease_name: str,
    sensor_data : dict | None = None,
) -> tuple[str, list[dict]]:
    """
    MEDIUM tier: Gemini 2.5 Flash via Google API.
    Seimbang antara kecepatan dan kualitas, context window 1M token.
    Returns: (answer_text, retrieved_chunks)
    """
    prompt, chunks = _build_rag_prompt(disease_name, sensor_data)
    text, _        = _gemini_generate(prompt)
    return text, chunks


def get_recommendation_high(
    disease_name: str,
    sensor_data : dict | None = None,
) -> tuple[str, list[dict]]:
    """
    HIGH tier: Llama 3.3-70B via Groq API (cloud).
    Model besar, kualitas tinggi, response cepat berkat hardware Groq.
    Returns: (answer_text, retrieved_chunks)
    """
    prompt, chunks = _build_rag_prompt(disease_name, sensor_data)
    text = _groq_generate(prompt)
    return text, chunks


# ═════════════════════════════════════════════════════════════════
# COMPARE — 3 Tier LLM (Low / Medium / High)
# ═════════════════════════════════════════════════════════════════
def compare_llm_recommendation(disease_name: str, sensor_data: dict | None = None) -> dict:
    """
    Membandingkan ketiga tier LLM (LOW / MEDIUM / HIGH) untuk satu kasus penyakit.
    Mengembalikan hasil lengkap termasuk waktu respons dan chunk RAG yang digunakan.
    """
    results = {}

    # ── LOW: Qwen2.5-3B (Ollama) ──────────────────────────────────
    try:
        t0             = time.perf_counter()
        answer, chunks = get_recommendation_low(disease_name, sensor_data)
        elapsed        = round(time.perf_counter() - t0, 3)
        results["low"] = {
            "tier"            : "LOW",
            "llm"             : "Ollama — Qwen2.5-3B (LOW)",
            "model"           : LLM_LOW_MODEL,
            "answer"          : answer,
            "response_time"   : elapsed,
            "status"          : "success",
            "rag_chunks_used" : len(chunks),
            "rag_top_scores"  : [round(c["score"], 4) for c in chunks[:3]],
        }
    except Exception as e:
        logger.error(f"LOW (Qwen 3B) error: {e}")
        results["low"] = {
            "tier"  : "LOW",
            "llm"   : "Ollama — Qwen2.5-3B (LOW)",
            "model" : LLM_LOW_MODEL,
            "answer": None,
            "status": f"error: {str(e)}",
        }

    # ── MEDIUM: Gemini 2.5 Flash ───────────────────────────────────
    try:
        t0                = time.perf_counter()
        answer, chunks    = get_recommendation_medium(disease_name, sensor_data)
        elapsed           = round(time.perf_counter() - t0, 3)
        model_used        = GEMINI_MODEL_NAME
        results["medium"] = {
            "tier"            : "MEDIUM",
            "llm"             : f"Gemini — {model_used} (MEDIUM)",
            "model"           : model_used,
            "answer"          : answer,
            "response_time"   : elapsed,
            "status"          : "success",
            "rag_chunks_used" : len(chunks),
            "rag_top_scores"  : [round(c["score"], 4) for c in chunks[:3]],
        }
    except Exception as e:
        logger.error(f"MEDIUM (Gemini) error: {e}")
        results["medium"] = {
            "tier"  : "MEDIUM",
            "llm"   : f"Gemini — {GEMINI_MODEL_NAME} (MEDIUM)",
            "model" : GEMINI_MODEL_NAME,
            "answer": None,
            "status": f"error: {str(e)}",
        }

    # ── HIGH: Llama3.3-70B (Groq API) ────────────────────────────
    try:
        t0              = time.perf_counter()
        answer, chunks  = get_recommendation_high(disease_name, sensor_data)
        elapsed         = round(time.perf_counter() - t0, 3)
        results["high"] = {
            "tier"            : "HIGH",
            "llm"             : "Groq — Llama3.3-70B (HIGH)",
            "model"           : LLM_HIGH_MODEL,
            "answer"          : answer,
            "response_time"   : elapsed,
            "status"          : "success",
            "rag_chunks_used" : len(chunks),
            "rag_top_scores"  : [round(c["score"], 4) for c in chunks[:3]],
        }
    except Exception as e:
        logger.error(f"HIGH (Llama 70B Groq) error: {e}")
        results["high"] = {
            "tier"  : "HIGH",
            "llm"   : "Groq — Llama3.3-70B (HIGH)",
            "model" : LLM_HIGH_MODEL,
            "answer": None,
            "status": f"error: {str(e)}",
        }

    successful = {k: v for k, v in results.items() if v.get("status") == "success"}
    fastest    = (
        min(successful, key=lambda k: successful[k]["response_time"])
        if successful else None
    )

    gemini_info = "tidak tersedia"
    if _gemini_working:
        gemini_info = f"{_gemini_working[0]} (API {_gemini_working[1]})"

    return {
        "disease_name"        : disease_name,
        "results"             : results,
        "fastest_llm"         : fastest,
        "sensor_used"         : sensor_data is not None,
        "rag_enabled"         : True,
        "gemini_model_active" : gemini_info,
        "ollama_url"          : OLLAMA_BASE_URL,
    }


# ═════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — fungsi lama tetap bisa dipakai
# ═════════════════════════════════════════════════════════════════

# Alias untuk kode lama yang pakai "groq" / "gemini"
def get_recommendation_groq_rag(disease_name: str, sensor_data: dict | None = None) -> tuple[str, list[dict]]:
    """Alias lama → sekarang pakai HIGH (Llama 70B, kualitas terbaik lokal)."""
    return get_recommendation_high(disease_name, sensor_data)


def get_recommendation_gemini_rag(disease_name: str, sensor_data: dict | None = None) -> tuple[str, list[dict]]:
    """Alias lama → sekarang pakai MEDIUM (Gemini API)."""
    return get_recommendation_medium(disease_name, sensor_data)


# Alias tanpa RAG — fallback ke prompt biasa
def get_recommendation_groq(disease_name: str, sensor_data: dict | None = None) -> str:
    """Alias lama → pakai Groq HIGH tanpa RAG."""
    prompt = _build_recommendation_prompt(disease_name, sensor_data)
    return _groq_generate(prompt)


def get_recommendation_gemini(disease_name: str, sensor_data: dict | None = None) -> str:
    """Alias lama → pakai Gemini MEDIUM tanpa RAG."""
    prompt  = _build_recommendation_prompt(disease_name, sensor_data)
    text, _ = _gemini_generate(prompt)
    return text


def get_chat_response_groq(question: str, disease_context: str) -> str:
    """Alias lama → pakai Groq HIGH untuk chat."""
    context = ""
    if disease_context and disease_context.strip():
        context = (
            f"\nKonteks: Petani sedang menangani penyakit "
            f"'{disease_context}' pada tanaman padinya.\n"
        )
    return _groq_generate(context + question)


def get_chat_response_gemini(question: str, disease_context: str) -> str:
    """Alias lama → pakai Gemini MEDIUM untuk chat."""
    context = ""
    if disease_context and disease_context.strip():
        context = (
            f"Konteks: Petani sedang menangani penyakit "
            f"'{disease_context}' pada tanaman padinya.\n\n"
        )
    text, _ = _gemini_generate(context + question)
    return text


def get_recommendation(disease_name: str, sensor_data: dict | None = None) -> str:
    """Default recommendation → MEDIUM (Gemini)."""
    return get_recommendation_gemini(disease_name, sensor_data)


def get_chat_response(question: str, disease_context: str) -> str:
    """Default chat → MEDIUM (Gemini)."""
    return get_chat_response_gemini(question, disease_context)