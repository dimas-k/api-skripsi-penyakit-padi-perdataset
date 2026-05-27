"""
llm.py — LLM dengan Integrasi RAG
====================================
Update dari versi sebelumnya:
- Ditambahkan fungsi RAG-aware: get_recommendation_with_rag()
- Fungsi lama tetap tersedia untuk backward compatibility
- Setiap panggilan rekomendasi kini menggunakan konteks dari knowledge base
"""

import os
import time
import logging
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


groq_client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if not _GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY belum diset di .env")

_GEMINI_BASE = "https://generativelanguage.googleapis.com"
_GEMINI_CANDIDATES = [
    ("gemini-2.5-flash",      "v1beta"),
    ("gemini-2.5-flash",      "v1"),
    ("gemini-2.0-flash",      "v1"),
    ("gemini-2.0-flash",      "v1beta"),
    ("gemini-2.0-flash-lite", "v1"),
    ("gemini-2.0-flash-lite", "v1beta"),
]

GEMINI_MODEL_NAME            = _GEMINI_CANDIDATES[0][0]
_gemini_working: tuple[str, str] | None = None

print(f"✅ Gemini siap (lazy) — akan mencoba {_GEMINI_CANDIDATES[0][0]} saat request pertama")


# ═════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Kamu adalah asisten pertanian yang membantu petani padi langsung di lapangan.

Gunakan bahasa yang sederhana, hangat, dan mudah dipahami oleh petani biasa.
Hindari istilah ilmiah yang rumit — kalau terpaksa pakai, jelaskan artinya dengan singkat.
Jawab langsung ke solusi yang bisa dikerjakan hari ini, bukan teori panjang.
Jika tersedia informasi dari basis pengetahuan, gunakan informasi tersebut sebagai acuan utama.

Penyakit padi yang kamu tangani:
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

Selalu jawab dalam Bahasa Indonesia."""


# ═════════════════════════════════════════════════════════════════
# INTERNAL: Gemini REST call (tidak berubah)
# ═════════════════════════════════════════════════════════════════
def _gemini_generate(prompt: str) -> tuple[str, str]:
    global GEMINI_MODEL_NAME, _gemini_working
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
# PROMPT BUILDER — Versi Lama (tanpa RAG, untuk backward compat)
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


# ═════════════════════════════════════════════════════════════════
# ✅ BARU: RAG-AWARE RECOMMENDATION
# ═════════════════════════════════════════════════════════════════
def _build_rag_prompt(disease_name: str, sensor_data: dict | None = None) -> tuple[str, list[dict]]:
    """
    Build prompt menggunakan RAG — konteks diambil dari knowledge base.
    Returns: (prompt, retrieved_chunks)
    """
    try:
        from rag import build_rag_prompt
        prompt, chunks = build_rag_prompt(
            disease_name=disease_name,
            sensor_data=sensor_data,
        )
        return prompt, chunks
    except ImportError:
        logger.warning("Modul rag.py tidak ditemukan, fallback ke prompt tanpa RAG.")
        return _build_recommendation_prompt(disease_name, sensor_data), []
    except Exception as e:
        logger.warning(f"RAG gagal ({e}), fallback ke prompt tanpa RAG.")
        return _build_recommendation_prompt(disease_name, sensor_data), []


# ─── Groq dengan RAG ──────────────────────────────────────────────
def get_recommendation_groq_rag(
    disease_name: str,
    sensor_data : dict | None = None,
) -> tuple[str, list[dict]]:
    """
    Generate rekomendasi dengan Groq + RAG.
    Returns: (answer_text, retrieved_chunks)
    """
    prompt, chunks = _build_rag_prompt(disease_name, sensor_data)
    response = groq_client.chat.completions.create(
        model    = GROQ_MODEL_NAME,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature = 0.7,
        max_tokens  = 2048,
    )
    return response.choices[0].message.content, chunks


# ─── Gemini dengan RAG ───────────────────────────────────────────
def get_recommendation_gemini_rag(
    disease_name: str,
    sensor_data : dict | None = None,
) -> tuple[str, list[dict]]:
    """
    Generate rekomendasi dengan Gemini + RAG.
    Returns: (answer_text, retrieved_chunks)
    """
    prompt, chunks = _build_rag_prompt(disease_name, sensor_data)
    text, _        = _gemini_generate(prompt)
    return text, chunks


# ═════════════════════════════════════════════════════════════════
# GROQ FUNCTIONS (backward compatible — tanpa RAG)
# ═════════════════════════════════════════════════════════════════
def get_recommendation_groq(disease_name: str, sensor_data: dict | None = None) -> str:
    prompt   = _build_recommendation_prompt(disease_name, sensor_data)
    response = groq_client.chat.completions.create(
        model    = GROQ_MODEL_NAME,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature = 0.7,
        max_tokens  = 2048,
    )
    return response.choices[0].message.content


def get_chat_response_groq(question: str, disease_context: str) -> str:
    context = ""
    if disease_context and disease_context.strip():
        context = (
            f"\nKonteks: Petani sedang menangani penyakit "
            f"'{disease_context}' pada tanaman padinya.\n"
        )
    response = groq_client.chat.completions.create(
        model    = GROQ_MODEL_NAME,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + context},
            {"role": "user",   "content": question},
        ],
        temperature = 0.7,
        max_tokens  = 2048,
    )
    return response.choices[0].message.content


# ═════════════════════════════════════════════════════════════════
# GEMINI FUNCTIONS (backward compatible — tanpa RAG)
# ═════════════════════════════════════════════════════════════════
def get_recommendation_gemini(disease_name: str, sensor_data: dict | None = None) -> str:
    prompt  = _build_recommendation_prompt(disease_name, sensor_data)
    text, _ = _gemini_generate(prompt)
    return text


def get_chat_response_gemini(question: str, disease_context: str) -> str:
    context = ""
    if disease_context and disease_context.strip():
        context = (
            f"Konteks: Petani sedang menangani penyakit "
            f"'{disease_context}' pada tanaman padinya.\n\n"
        )
    text, _ = _gemini_generate(context + question)
    return text


# ═════════════════════════════════════════════════════════════════
# COMPARE — Groq vs Gemini (dengan RAG)
# ═════════════════════════════════════════════════════════════════
def compare_llm_recommendation(disease_name: str, sensor_data: dict | None = None) -> dict:
    results = {}

    # ── Groq + RAG ────────────────────────────────────────────────
    try:
        t0                  = time.perf_counter()
        groq_answer, chunks = get_recommendation_groq_rag(disease_name, sensor_data)
        groq_time           = round(time.perf_counter() - t0, 3)
        results["groq"] = {
            "llm"              : "Groq — LLaMA 3.3 70B Versatile",
            "model"            : GROQ_MODEL_NAME,
            "answer"           : groq_answer,
            "response_time"    : groq_time,
            "status"           : "success",
            "rag_chunks_used"  : len(chunks),
            "rag_top_scores"   : [round(c["score"], 4) for c in chunks[:3]],
        }
    except Exception as e:
        logger.error(f"Groq error: {e}")
        results["groq"] = {
            "llm"    : "Groq — LLaMA 3.3 70B Versatile",
            "model"  : GROQ_MODEL_NAME,
            "answer" : None,
            "status" : f"error: {str(e)}",
        }

    # ── Gemini + RAG ──────────────────────────────────────────────
    try:
        t0                    = time.perf_counter()
        gemini_answer, chunks = get_recommendation_gemini_rag(disease_name, sensor_data)
        gemini_time           = round(time.perf_counter() - t0, 3)
        model_used            = GEMINI_MODEL_NAME
        results["gemini"] = {
            "llm"              : f"Google — Gemini ({model_used})",
            "model"            : model_used,
            "answer"           : gemini_answer,
            "response_time"    : gemini_time,
            "status"           : "success",
            "rag_chunks_used"  : len(chunks),
            "rag_top_scores"   : [round(c["score"], 4) for c in chunks[:3]],
        }
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        results["gemini"] = {
            "llm"    : f"Google — Gemini ({GEMINI_MODEL_NAME})",
            "model"  : GEMINI_MODEL_NAME,
            "answer" : None,
            "status" : f"error: {str(e)}",
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
    }


# ── Backward-compat aliases ──────────────────────────────────────
def get_recommendation(disease_name: str, sensor_data: dict | None = None) -> str:
    return get_recommendation_groq(disease_name, sensor_data)

def get_chat_response(question: str, disease_context: str) -> str:
    return get_chat_response_groq(question, disease_context)
