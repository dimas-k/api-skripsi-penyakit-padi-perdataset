"""
database.py — Supabase client & semua operasi database

Tabel:
  - users         : identitas perangkat pengguna (device_id based, no login)
  - predictions   : riwayat setiap deteksi penyakit
  - chat_messages : riwayat percakapan chatbot
"""

import os
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ═════════════════════════════════════════════════════════════════
# CLIENT (singleton)
# ═════════════════════════════════════════════════════════════════
_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL dan SUPABASE_KEY harus diisi di file .env"
            )
        _client = create_client(url, key)
    return _client


# ═════════════════════════════════════════════════════════════════
# USERS
# ═════════════════════════════════════════════════════════════════

def get_or_create_user(device_id: str, device_info: Optional[dict] = None) -> dict:
    """
    Ambil user berdasarkan device_id.
    Jika belum ada, buat baru (upsert).
    Dipanggil setiap kali ada request dari perangkat baru.
    """
    db = get_client()

    # Cek apakah user sudah ada
    result = (
        db.table("users")
        .select("*")
        .eq("device_id", device_id)
        .execute()
    )

    if result.data:
        # Sudah ada — update last_seen
        user = result.data[0]
        db.table("users").update({"last_seen": "NOW()"}).eq("id", user["id"]).execute()
        return user

    # Belum ada — buat baru
    new_user = {
        "device_id"  : device_id,
        "device_info": device_info or {},
    }
    insert_result = db.table("users").insert(new_user).execute()
    return insert_result.data[0] if insert_result.data else {}


def get_user_by_device(device_id: str) -> Optional[dict]:
    """Ambil data user berdasarkan device_id."""
    db     = get_client()
    result = (
        db.table("users")
        .select("*")
        .eq("device_id", device_id)
        .execute()
    )
    return result.data[0] if result.data else None


# ═════════════════════════════════════════════════════════════════
# PREDICTIONS
# ═════════════════════════════════════════════════════════════════

def save_prediction(
    prediction_id    : str,
    user_id          : str,
    predicted_class  : str,
    confidence       : float,
    detection_time_ms: float,
    recommendation   : str,
    llm_used         : str,
    sensor_used      : bool,
    sensor_data      : Optional[dict] = None,
    swin_results     : Optional[dict] = None,
    vote_method      : Optional[str]  = None,
    majority_count   : Optional[str]  = None,
) -> dict:
    """
    Simpan satu hasil prediksi ke tabel `predictions`.
    Trigger otomatis akan update last_seen & total_predictions di tabel users.
    """
    db   = get_client()
    data = {
        "id"               : prediction_id,
        "user_id"          : user_id,
        "predicted_class"  : predicted_class,
        "confidence"       : confidence,
        "detection_time_ms": detection_time_ms,
        "recommendation"   : recommendation,
        "llm_used"         : llm_used,
        "sensor_used"      : sensor_used,
        "sensor_data"      : sensor_data,
        "swin_results"     : swin_results,
        "vote_method"      : vote_method,
        "majority_count"   : majority_count,
    }
    result = db.table("predictions").insert(data).execute()
    return result.data[0] if result.data else {}


def get_predictions(
    user_id: str,
    limit  : int = 20,
    offset : int = 0,
) -> tuple[list, int]:
    """
    Ambil riwayat prediksi berdasarkan user_id.
    Mengembalikan (list_of_rows, total_count).
    """
    db = get_client()

    count_result = (
        db.table("predictions")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .execute()
    )
    total = count_result.count or 0

    result = (
        db.table("predictions")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data or [], total


def delete_prediction(prediction_id: str) -> bool:
    """Hapus satu prediksi berdasarkan ID."""
    db     = get_client()
    result = db.table("predictions").delete().eq("id", prediction_id).execute()
    return len(result.data) > 0


def get_prediction_by_id(prediction_id: str) -> Optional[dict]:
    """Ambil detail satu prediksi termasuk swin_results dan sensor_data."""
    db = get_client()
    try:
        result = (
            db.table("predictions")
            .select("*")
            .eq("id", prediction_id)
            .single()
            .execute()
        )
        return result.data
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════
# CHAT MESSAGES
# ═════════════════════════════════════════════════════════════════

def save_chat_message(
    user_id        : str,
    question       : str,
    answer         : str,
    disease_context: str,
    llm_used       : str,
    prediction_id  : Optional[str] = None,
) -> dict:
    """
    Simpan satu pesan chat ke tabel `chat_messages`.
    Terhubung ke users via user_id, dan opsional ke predictions via prediction_id.
    """
    db   = get_client()
    data = {
        "user_id"        : user_id,
        "prediction_id"  : prediction_id,
        "question"       : question,
        "answer"         : answer,
        "disease_context": disease_context,
        "llm_used"       : llm_used,
    }
    result = db.table("chat_messages").insert(data).execute()
    return result.data[0] if result.data else {}


def get_chat_messages(user_id: str, limit: int = 50) -> list:
    """Ambil riwayat chat berdasarkan user_id, terbaru duluan."""
    db     = get_client()
    result = (
        db.table("chat_messages")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_chat_by_prediction(prediction_id: str) -> list:
    """Ambil semua pesan chat terkait satu prediksi."""
    db     = get_client()
    result = (
        db.table("chat_messages")
        .select("*")
        .eq("prediction_id", prediction_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data or []


# ═════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═════════════════════════════════════════════════════════════════

def test_connection() -> dict:
    """Cek koneksi ke Supabase — dipanggil saat startup dan GET /db-test."""
    try:
        db     = get_client()
        result = db.table("users").select("id").limit(1).execute()
        return {
            "status" : "connected",
            "message": "Supabase terhubung dengan baik ✅",
        }
    except ValueError as e:
        return {"status": "misconfigured", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Gagal terhubung: {str(e)}"}
