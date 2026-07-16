import os
import time
import uuid
import random
from collections import Counter
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from schemas import (
    PredictionResponse, SwingModelResult, ChatResponse,
    HistoryResponse, HistoryItem, ChatHistoryItem, ChatHistoryResponse,
    Pagination, SensorResponse, SensorStatus,
    CompareResponse, ModelCompareResult, DetectionTimeStats,
    UserResponse,
)
from model import load_model, load_all_models, load_gabungan_models, predict, ARCH_LABELS, DATASET_LABELS
from ood import load_ood_stats, check_ood
from llm import (
    get_recommendation_groq_rag, get_recommendation_gemini_rag,
    get_chat_response_groq, get_chat_response_gemini,
    compare_llm_recommendation,
)
import database as db
import thingsboard as tb


# ═════════════════════════════════════════════════════════════════
# MODEL STORE
# ═════════════════════════════════════════════════════════════════
ml_model           = {}
ml_models          = {}
ml_models_gabungan = {}   # 5 arsitektur dilatih di dataset gabungan (untuk /compare/gabungan)


# ═════════════════════════════════════════════════════════════════
# HELPER — resolve user_id dari device_id
# Dipanggil di setiap endpoint yang butuh user_id
# ═════════════════════════════════════════════════════════════════
def _resolve_user(device_id: str, device_info: Optional[dict] = None) -> Optional[str]:
    """
    Ambil atau buat user berdasarkan device_id.
    Return user UUID, atau None jika Supabase tidak terhubung.
    """
    if not device_id or device_id == "anonymous":
        return None
    try:
        user = db.get_or_create_user(device_id, device_info)
        return user.get("id")
    except Exception as e:
        print(f"⚠️  Gagal resolve user ({device_id}): {e}")
        return None


# ═════════════════════════════════════════════════════════════════
# HELPER — weighted voting
# ═════════════════════════════════════════════════════════════════
def _weighted_vote(
    predictions       : list[tuple[str, float]],
    min_confidence    : float = 60.0,
    high_conf_threshold: float = 85.0,
) -> tuple[str, dict, str]:
    if not predictions:
        return "unknown", {}, "fallback_count"

    weight_map: dict[str, float] = {}
    for cls, conf in predictions:
        weight_map[cls] = round(weight_map.get(cls, 0) + conf, 2)

    high_conf_preds = [(cls, conf) for cls, conf in predictions if conf >= high_conf_threshold]
    if high_conf_preds:
        high_classes = set(cls for cls, _ in high_conf_preds)
        if len(high_classes) == 1:
            return high_conf_preds[0][0], weight_map, "high_confidence_override"
        high_weight: dict[str, float] = {}
        for cls, conf in high_conf_preds:
            high_weight[cls] = round(high_weight.get(cls, 0) + conf, 2)
        return max(high_weight, key=lambda k: high_weight[k]), weight_map, "high_confidence_tiebreak"

    filtered = [(cls, conf) for cls, conf in predictions if conf >= min_confidence]
    if filtered:
        w: dict[str, float] = {}
        for cls, conf in filtered:
            w[cls] = round(w.get(cls, 0) + conf, 2)
        return max(w, key=lambda k: w[k]), weight_map, "weighted"

    return max(weight_map, key=lambda k: weight_map[k]), weight_map, "weighted_no_threshold"


def _parse_model_key(model_key: str) -> tuple[str, str]:
    if "__" in model_key:
        arch, dataset = model_key.split("__", 1)
        return arch, dataset
    for arch in ARCH_LABELS:
        if model_key.startswith(arch + "_"):
            return arch, model_key[len(arch) + 1:]
    return model_key, "unknown"


# ═════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n══ Mengecek koneksi Supabase ══")
    db_status = db.test_connection()
    print(f"{'✅' if db_status['status'] == 'connected' else '⚠️ '} {db_status['message']}")

    print("\n══ Loading model GABUNGAN utama (/predict) ══")
    try:
        ml_model["model"], ml_model["class_names"], ml_model["model_name"] = load_model()
        print(f"✅ Model utama siap: {ml_model['model_name']}")
    except FileNotFoundError as e:
        print(f"⚠️  Model utama tidak ditemukan: {e}")

    all_loaded = load_all_models()
    for model_key, meta in all_loaded.items():
        if meta["status"] == "loaded":
            ml_models[model_key] = meta
    print(f"✅ Total model siap: {len(ml_models)} / {len(all_loaded)}\n")

    # ── Muat 5 model DATASET GABUNGAN (untuk /compare/gabungan) ───
    print("══ Loading model DATASET GABUNGAN (5 arsitektur) ══")
    try:
        gab_loaded = load_gabungan_models()
        for model_key, meta in gab_loaded.items():
            if meta["status"] == "loaded":
                ml_models_gabungan[model_key] = meta
        print(f"✅ Model gabungan siap: {len(ml_models_gabungan)} / {len(gab_loaded)}\n")
    except Exception as e:
        print(f"⚠️  Gagal memuat model gabungan (diabaikan, API tetap jalan): {e}")

    # ── Muat statistik OOD (deteksi "bukan padi") ─────────────────
    print("══ Memuat statistik OOD (gate bukan-padi) ══")
    try:
        load_ood_stats()
    except Exception as e:
        # Gate OOD tidak boleh menggagalkan startup API (fail-open).
        print(f"⚠️  Gagal memuat OOD (diabaikan, API tetap jalan): {e}")

    yield
    ml_model.clear()
    ml_models.clear()
    ml_models_gabungan.clear()


# ═════════════════════════════════════════════════════════════════
# APP
# ═════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "Paddy Disease Detection API",
    description = (
        "API deteksi penyakit daun padi — 5 arsitektur × 4 dataset = 20 model. "
        "Rekomendasi LLM: Groq (LLaMA 3.3 70B) & Gemini. "
        "Database: Supabase (PostgreSQL) — 3 tabel: users, predictions, chat_messages."
    ),
    version  = "4.1.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════════
# SENSOR
# ═════════════════════════════════════════════════════════════════
def _generate_sensor_data() -> dict:
    return {
        "suhu_udara"        : round(random.uniform(28.0, 34.0), 1),
        "kelembaban_udara"  : round(random.uniform(70.0, 92.0), 1),
        "suhu_tanah"        : round(random.uniform(24.0, 30.0), 1),
        "kelembaban_tanah"  : round(random.uniform(55.0, 85.0), 1),
        "ph_tanah"          : round(random.uniform(5.5, 7.2),   1),
        "nitrogen"          : round(random.uniform(10.0, 50.0), 1),
        "fosfor"            : round(random.uniform(5.0,  30.0), 1),
        "kalium"            : round(random.uniform(80.0, 200.0), 1),
        "intensitas_cahaya" : round(random.uniform(15000, 60000), 0),
        "curah_hujan"       : round(random.uniform(0.0,  25.0), 1),
    }


def _evaluate_sensor(data: dict) -> list[SensorStatus]:
    thresholds = {
        "suhu_udara"        : {"satuan": "°C",     "min": 20,    "max": 35,    "label": "Suhu Udara"},
        "kelembaban_udara"  : {"satuan": "%",       "min": 60,    "max": 90,    "label": "Kelembaban Udara"},
        "suhu_tanah"        : {"satuan": "°C",     "min": 20,    "max": 30,    "label": "Suhu Tanah"},
        "kelembaban_tanah"  : {"satuan": "%",       "min": 50,    "max": 80,    "label": "Kelembaban Tanah"},
        "ph_tanah"          : {"satuan": "",        "min": 5.5,   "max": 7.0,   "label": "pH Tanah"},
        "nitrogen"          : {"satuan": "mg/kg",  "min": 15,    "max": 45,    "label": "Nitrogen (N)"},
        "fosfor"            : {"satuan": "mg/kg",  "min": 8,     "max": 25,    "label": "Fosfor (P)"},
        "kalium"            : {"satuan": "mg/kg",  "min": 100,   "max": 180,   "label": "Kalium (K)"},
        "intensitas_cahaya" : {"satuan": "lux",     "min": 20000, "max": 55000, "label": "Intensitas Cahaya"},
        "curah_hujan"       : {"satuan": "mm/hari","min": 0,     "max": 20,    "label": "Curah Hujan"},
    }
    info_map = {
        "suhu_udara"        : ("Mendukung fotosintesis optimal",   "Terlalu dingin, pertumbuhan lambat",  "Terlalu panas, stres tanaman"),
        "kelembaban_udara"  : ("Kelembaban ideal",                 "Terlalu kering, rentan Blas",         "Terlalu lembab, rentan jamur"),
        "suhu_tanah"        : ("Suhu tanah ideal",                 "Dingin, akar kurang aktif",           "Panas, ganggu penyerapan air"),
        "kelembaban_tanah"  : ("Kelembaban tanah cukup",           "Kering, perlu pengairan",             "Terlalu basah, rentan busuk akar"),
        "ph_tanah"          : ("pH optimal untuk padi",            "Asam, hambat penyerapan unsur hara",  "Basa, hambat penyerapan Fe & Mn"),
        "nitrogen"          : ("Nitrogen cukup",                   "Nitrogen rendah, daun menguning",     "Nitrogen berlebih, rentan Hawar"),
        "fosfor"            : ("Fosfor cukup",                     "Fosfor rendah, akar lemah",           "Fosfor berlebih, ganggu penyerapan Zn"),
        "kalium"            : ("Kalium cukup",                     "Kalium rendah, rentan penyakit",      "Kalium berlebih, hambat penyerapan Ca"),
        "intensitas_cahaya" : ("Cahaya optimal",                   "Kurang cahaya, pertumbuhan lambat",   "Terlalu terik, stres panas"),
        "curah_hujan"       : ("Curah hujan ideal",                "Kering, perlu irigasi tambahan",      "Hujan berlebih, rentan Hawar & Blas"),
    }
    statuses = []
    for key, thresh in thresholds.items():
        nilai  = data.get(key, 0)
        mn, mx = thresh["min"], thresh["max"]
        ok, lo, hi = info_map[key]
        if nilai < mn:   status, ket = "rendah", lo
        elif nilai > mx: status, ket = "tinggi", hi
        else:            status, ket = "normal", ok
        statuses.append(SensorStatus(
            parameter=thresh["label"], nilai=nilai,
            satuan=thresh["satuan"], status=status, keterangan=ket,
        ))
    return statuses


# ═════════════════════════════════════════════════════════════════
# GENERAL
# ═════════════════════════════════════════════════════════════════
@app.get("/health", tags=["General"])
async def health_check():
    by_arch = {}
    for key in ml_models:
        arch, _ = _parse_model_key(key)
        by_arch[arch] = by_arch.get(arch, 0) + 1
    return {
        "status"        : "healthy",
        "version"       : "4.1.0",
        "model_utama"   : ml_model.get("model_name", "belum_load"),
        "total_model"   : len(ml_models),
        "model_per_arch": by_arch,
        "llm_groq"      : "llama-3.3-70b-versatile",
        "llm_gemini"    : "gemini-2.0-flash",
        "num_classes"   : len(ml_model.get("class_names", [])),
        "database"      : db.test_connection(),
    }


@app.get("/classes", tags=["General"])
async def get_classes():
    return {"classes": ml_model.get("class_names", []), "total": len(ml_model.get("class_names", []))}


@app.get("/models", tags=["General"])
async def list_models():
    result = {}
    for key, meta in ml_models.items():
        result[key] = {
            "arsitektur": meta["arch_label"],
            "dataset"   : meta["dataset_label"],
            "model_name": meta["model_name"],
            "path"      : meta.get("path"),
        }
    return {"total_model": len(result), "models": result}


@app.get("/db-test", tags=["General"])
async def db_test():
    result = db.test_connection()
    if result["status"] != "connected":
        raise HTTPException(status_code=503, detail=result["message"])
    return result


# ═════════════════════════════════════════════════════════════════
# USERS
# ═════════════════════════════════════════════════════════════════
@app.post("/users/register", response_model=UserResponse, tags=["Users"])
async def register_device(
    x_user_id    : Optional[str] = Header(None),
    x_device_info: Optional[str] = Header(None),
):
    """
    Daftarkan perangkat ke database.
    Dipanggil satu kali saat aplikasi pertama dibuka.
    Jika device_id sudah terdaftar, data yang ada dikembalikan (tidak duplikat).

    Header:
    - x-user-id     : device UUID unik dari HP (wajib)
    - x-device-info : info perangkat dalam format JSON string (opsional)
    """
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Header x-user-id wajib diisi")

    import json
    device_info = {}
    if x_device_info:
        try:
            device_info = json.loads(x_device_info)
        except Exception:
            device_info = {"raw": x_device_info}

    try:
        user = db.get_or_create_user(x_user_id, device_info)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gagal registrasi: {str(e)}")

    return UserResponse(
        user_id          = user["id"],
        device_id        = user["device_id"],
        total_predictions= user.get("total_predictions", 0),
        first_seen       = user.get("first_seen", ""),
        last_seen        = user.get("last_seen", ""),
    )


@app.get("/users/{device_id}", response_model=UserResponse, tags=["Users"])
async def get_user_info(device_id: str):
    """Ambil info user berdasarkan device_id."""
    try:
        user = db.get_user_by_device(device_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    return UserResponse(
        user_id          = user["id"],
        device_id        = user["device_id"],
        total_predictions= user.get("total_predictions", 0),
        first_seen       = user.get("first_seen", ""),
        last_seen        = user.get("last_seen", ""),
    )


# ════════════════════════���════════════════════════════════════════
# SENSOR
# ═════════════════════════════════════════════════════════════════
@app.get("/sensor", response_model=SensorResponse, tags=["Sensor"])
async def get_sensor_data(source: str = "auto"):
    """
    Data sensor Stasiun AWS.
    - source=auto (default): ambil data ASLI ThingsBoard, fallback dummy bila gagal.
    - source=thingsboard   : paksa ThingsBoard (error 502 bila gagal).
    - source=dummy         : paksa data contoh.
    """
    if source in ("auto", "thingsboard"):
        try:
            device_id = tb.resolve_device_id()
            latest    = tb.get_average_values(device_id)   # rata-rata periode data (bukan nilai terakhir)
            data, detail, abnormal = tb.build_snapshot(latest)
            if data:
                statuses = [
                    SensorStatus(
                        parameter=d["parameter"], nilai=d["nilai"], satuan=d["satuan"],
                        status=d["status"], keterangan=d["keterangan"],
                    )
                    for d in detail
                ]
                kesimpulan = (
                    "✅ Semua parameter sensor dalam batas normal."
                    if not abnormal else
                    f"⚠️ {len(abnormal)} parameter di luar batas normal: "
                    + ", ".join(abnormal) + ". Perlu perhatian lebih lanjut."
                )
                try:
                    periode = tb.periode_text(tb.get_data_coverage(device_id))
                except tb.ThingsBoardError:
                    periode = ""
                return SensorResponse(
                    lokasi="Stasiun AWS — ThingsBoard (PetaniTech)",
                    timestamp=tb.latest_timestamp_iso(latest) or datetime.now().isoformat(),
                    data=data, detail_status=statuses, kesimpulan=kesimpulan,
                    is_realtime=False, periode_data=(periode or None), sumber="thingsboard",
                    data_llm=tb.to_llm_sensor_dict(data),
                )
        except tb.ThingsBoardError as e:
            if source == "thingsboard":
                raise HTTPException(status_code=502, detail=str(e))
            print(f"⚠️  ThingsBoard tidak tersedia, pakai data dummy: {e}")

    # Fallback: data dummy
    data     = _generate_sensor_data()
    statuses = _evaluate_sensor(data)
    abnormal = [s for s in statuses if s.status != "normal"]
    kesimpulan = (
        "✅ Semua kondisi sensor dalam batas normal. Tanaman dalam kondisi baik."
        if not abnormal else
        f"⚠️ {len(abnormal)} parameter di luar batas normal: "
        + ", ".join(s.parameter for s in abnormal) + ". Perlu perhatian lebih lanjut."
    )
    return SensorResponse(
        lokasi="Sawah Demo (dummy) — Indramayu, Jawa Barat",
        timestamp=datetime.now().isoformat(),
        data=data, detail_status=statuses, kesimpulan=kesimpulan,
        is_realtime=False, periode_data=None, sumber="dummy",
    )


def _resolve_sensor_for_llm(use_sensor: bool, manual_sensor: Optional[str]) -> Optional[dict]:
    """
    Tentukan data sensor untuk konteks LLM:
    1. Input MANUAL dari pengguna (realtime) — field boleh kosong sebagian.
    2. Data ThingsBoard (historis) bila use_sensor=True.
    3. None bila tidak ada.
    """
    import json as _json
    if manual_sensor:
        try:
            manual = _json.loads(manual_sensor)
        except Exception:
            manual = {}
        if isinstance(manual, dict):
            cleaned = {k: v for k, v in manual.items() if v not in (None, "", [])}
            if cleaned:
                return cleaned
    if use_sensor:
        try:
            device_id = tb.resolve_device_id()
            latest = tb.get_average_values(device_id)   # rata-rata periode data (bukan nilai terakhir)
            data, _, _ = tb.build_snapshot(latest)
            llm_dict = tb.to_llm_sensor_dict(data)
            if llm_dict:
                return llm_dict
        except tb.ThingsBoardError as e:
            print(f"⚠️  Sensor ThingsBoard gagal, pakai data dummy untuk LLM: {e}")
        # Fallback dummy agar KONSISTEN dengan endpoint /sensor (panel) yang juga
        # jatuh ke dummy bila ThingsBoard gagal. Dengan begini LLM tetap dapat
        # data sensor, tidak lagi menjawab "tidak ada informasi sensor".
        return _generate_sensor_data()
    return None


# ─── ThingsBoard (data sensor asli via time series) ───────────────
# Endpoint discovery untuk menemukan device & key telemetry aslinya.
# Setelah key diketahui, /sensor akan dipetakan ke data asli ini.
@app.get("/sensor/tb/devices", tags=["Sensor"])
async def tb_list_devices(text_search: str = ""):
    """Daftar device pada tenant ThingsBoard (untuk menemukan Device AWS)."""
    try:
        return {"devices": tb.list_devices(text_search)}
    except tb.ThingsBoardError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/sensor/tb/keys", tags=["Sensor"])
async def tb_list_keys():
    """Daftar key telemetry time-series pada device terkonfigurasi."""
    try:
        device_id = tb.resolve_device_id()
        return {"device_id": device_id, "keys": tb.get_timeseries_keys(device_id)}
    except tb.ThingsBoardError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/sensor/tb/timeseries", tags=["Sensor"])
async def tb_timeseries(
    keys: str = "",
    start: str = "",
    end: str = "",
    interval: int = 3600000,
    agg: str = "AVG",
    limit: int = 200,
):
    """
    Data TIME SERIES (riwayat) dari ThingsBoard — untuk grafik.
    - start/end: ISO datetime WIB, mis. 2025-01-01T00:00:00+07:00.
      Kosongkan untuk memakai rentang default sepanjang 2025.
    - interval: ukuran bucket agregasi (ms). 3600000 = 1 jam.
    - agg: NONE | AVG | MIN | MAX | SUM | COUNT.
    """
    try:
        device_id = tb.resolve_device_id()
        key_list = [k.strip() for k in keys.split(",") if k.strip()] or None
        return {
            "device_id": device_id,
            "timeseries": tb.get_timeseries(
                device_id, keys=key_list,
                start=start or None, end=end or None,
                interval=interval, agg=agg, limit=limit,
            ),
        }
    except tb.ThingsBoardError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/sensor/tb/latest", tags=["Sensor"])
async def tb_latest(keys: str = ""):
    """Nilai terakhir tiap key (endpoint latest, aman saat sensor non-aktif)."""
    try:
        device_id = tb.resolve_device_id()
        key_list = [k.strip() for k in keys.split(",") if k.strip()] or None
        return {
            "device_id": device_id,
            "latest": tb.get_latest_values(device_id, keys=key_list),
        }
    except tb.ThingsBoardError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ═════════════════════════════════════════════════════════════════
# DETECTION — /predict
# ═════════════════════════════════════════════════════════════════
@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict_disease(
    file         : UploadFile    = File(...),
    x_user_id    : Optional[str] = Header(None),
    x_device_info: Optional[str] = Header(None),
    use_sensor   : bool          = Form(False),
    manual_sensor: Optional[str] = Form(None),
    llm          : str           = Form("groq"),
):
    """
    Upload gambar → prediksi MODEL GABUNGAN (Swin-B, 14 kelas) → rekomendasi LLM+RAG
    → **simpan ke Supabase** (tabel users + predictions).

    Header:
    - x-user-id     : device UUID dari HP
    - x-device-info : info perangkat (JSON string, opsional)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    # ── Resolve user ──────────────────────────────────────────────
    import json
    device_info = {}
    if x_device_info:
        try:
            device_info = json.loads(x_device_info)
        except Exception:
            pass
    user_id = _resolve_user(x_user_id or "anonymous", device_info)

    # ── Gunakan MODEL GABUNGAN (deployment: 1 model, 14 kelas) ────
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model gabungan belum siap")
    swin_models = {"swin_base__gabungan": {
        "model": ml_model["model"], "class_names": ml_model["class_names"],
        "model_name": ml_model["model_name"], "arch_key": "swin_base",
        "dataset_label": "Dataset Gabungan (14 kelas)",
    }}

    # ── GATE OOD: tolak gambar yang BUKAN daun/tanaman padi ───────
    # Dicek SEBELUM diagnosa & LLM supaya benda random (mouse, wajah, dll)
    # tidak dipaksa jadi label penyakit. Jika stats OOD belum dimuat,
    # check_ood mengembalikan is_ood=False (fail-open) -> alur normal.
    ood = check_ood(image_bytes, ml_model["model"], ml_model.get("model_name", "swin_base"))
    if ood["is_ood"]:
        print(f"🚫 OOD terdeteksi (skor={ood['score']} > {ood['threshold']}, "
              f"terdekat={ood['nearest']}) — gambar ditolak sebagai bukan padi.")
        return PredictionResponse(
            predicted_class       = "bukan_padi",
            confidence_percentage = 0.0,
            detection_time_ms     = 0.0,
            recommendation        = (
                "Gambar yang diunggah sepertinya **bukan daun atau tanaman padi**, "
                "sehingga sistem tidak dapat mendiagnosis penyakit. \n\n"
                "Silakan ambil ulang foto **daun padi** yang jelas, fokus, dan "
                "cukup cahaya, lalu coba lagi."
            ),
            prediction_id         = str(uuid.uuid4()),
            saved_to_database     = False,
            total_swin_models     = 1,
            successful_models     = 0,
            vote_method           = "ood_rejected",
            majority_count        = "0 / 1 model (ditolak: bukan padi)",
        )

    # ── Prediksi ──────────────────────────────────────────────────
    swin_results   : dict[str, SwingModelResult] = {}
    pred_with_conf : list[tuple[str, float]]     = []
    total_time_ms  : float = 0.0

    for model_key, meta in swin_models.items():
        try:
            t0 = time.perf_counter()
            predicted_class_i, confidence_i = predict(
                image_bytes=image_bytes, model=meta["model"],
                class_names=meta["class_names"], model_name=meta["model_name"],
            )
            elapsed_ms     = round((time.perf_counter() - t0) * 1000, 2)
            total_time_ms += elapsed_ms
            swin_results[model_key] = SwingModelResult(
                dataset=meta["dataset_label"], predicted_class=predicted_class_i,
                confidence_percentage=confidence_i, detection_time_ms=elapsed_ms, status="success",
            )
            pred_with_conf.append((predicted_class_i, confidence_i))
        except Exception as e:
            swin_results[model_key] = SwingModelResult(
                dataset=meta.get("dataset_label", model_key),
                predicted_class=None, confidence_percentage=None,
                detection_time_ms=None, status=f"error: {str(e)}",
            )

    if not pred_with_conf:
        raise HTTPException(status_code=500, detail="Semua model Swin gagal")

    # ── Voting ────────────────────────────────────────────────────
    majority_class, weight_detail, vote_method = _weighted_vote(pred_with_conf)
    majority_count = sum(1 for cls, _ in pred_with_conf if cls == majority_class)

    sukses         = [v for v in swin_results.values() if v.status == "success"]
    avg_confidence = round(sum(v.confidence_percentage for v in sukses) / len(sukses), 2)
    avg_time_ms    = round(sum(v.detection_time_ms for v in sukses) / len(sukses), 2)
    best_swin_key  = max(
        (k for k, v in swin_results.items() if v.status == "success"),
        key=lambda k: swin_results[k].confidence_percentage, default=None,
    )
    best_swin_detail = swin_results[best_swin_key] if best_swin_key else None
    best_confidence  = best_swin_detail.confidence_percentage if best_swin_detail else avg_confidence

    # ── Sensor & LLM ──────────────────────────────────────────────
    # Prioritas: input manual (realtime) > ThingsBoard (historis) > tanpa sensor.
    sensor_data = _resolve_sensor_for_llm(use_sensor, manual_sensor)
    if llm == "gemini":
        recommendation, _ = get_recommendation_gemini_rag(majority_class, sensor_data)
        llm_used           = "gemini"
    else:
        recommendation, _ = get_recommendation_groq_rag(majority_class, sensor_data)
        llm_used           = "groq"

    # ── Simpan ke Supabase ────────────────────────────────────────
    prediction_id = str(uuid.uuid4())
    saved_ok      = False

    if user_id:
        try:
            # Upload gambar ke Supabase Storage supaya bisa tampil di riwayat.
            image_url = None
            try:
                image_url = db.upload_prediction_image(
                    prediction_id, image_bytes, file.content_type,
                )
            except Exception as e:
                print(f"⚠️  Gagal upload gambar (diabaikan): {e}")

            swin_dict = {k: v.model_dump() for k, v in swin_results.items()}
            db.save_prediction(
                prediction_id    = prediction_id,
                user_id          = user_id,
                predicted_class  = majority_class,
                confidence       = best_confidence,
                detection_time_ms= round(total_time_ms, 2),
                recommendation   = recommendation,
                llm_used         = llm_used,
                sensor_used      = use_sensor,
                sensor_data      = sensor_data,
                swin_results     = swin_dict,
                vote_method      = vote_method,
                majority_count   = f"{majority_count} / {len(pred_with_conf)} model",
                image_url        = image_url,
            )
            saved_ok = True
        except Exception as e:
            print(f"⚠️  Gagal simpan prediksi: {e}")

    return PredictionResponse(
        predicted_class=majority_class, confidence_percentage=best_confidence,
        detection_time_ms=round(total_time_ms, 2), recommendation=recommendation,
        prediction_id=prediction_id, saved_to_database=saved_ok,
        swin_results=swin_results, total_swin_models=len(swin_models),
        successful_models=len(sukses), vote_detail=weight_detail,
        vote_method=vote_method,
        majority_count=f"{majority_count} / {len(pred_with_conf)} model pilih kelas ini",
        avg_confidence=avg_confidence, avg_detection_time_ms=avg_time_ms,
        best_swin_model={"model_key": best_swin_key, "detail": best_swin_detail.model_dump() if best_swin_detail else None},
    )


# ═════════════════════════════════════════════════════════════����══
# DETECTION — /predict/{model_key}
# ═══════════════════════════════════════════════��═════════════════
@app.post("/predict/{model_key}", tags=["Detection"])
async def predict_with_model(
    model_key  : str,
    file       : UploadFile    = File(...),
    use_sensor : bool          = False,
    llm        : str           = "groq",
    x_user_id  : Optional[str] = Header(None),
):
    if model_key not in ml_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' tidak tersedia. Lihat GET /models.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    m  = ml_models[model_key]
    t0 = time.perf_counter()
    predicted_class, confidence = predict(
        image_bytes=image_bytes, model=m["model"],
        class_names=m["class_names"], model_name=m["model_name"],
    )
    detection_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    sensor_data = _generate_sensor_data() if use_sensor else None
    if llm == "gemini":
        recommendation, _ = get_recommendation_gemini_rag(predicted_class, sensor_data)
        llm_used           = "gemini"
    else:
        recommendation, _ = get_recommendation_groq_rag(predicted_class, sensor_data)
        llm_used           = "groq"

    prediction_id = str(uuid.uuid4())
    saved_ok      = False
    user_id       = _resolve_user(x_user_id or "anonymous")

    if user_id:
        try:
            db.save_prediction(
                prediction_id=prediction_id, user_id=user_id,
                predicted_class=predicted_class, confidence=confidence,
                detection_time_ms=detection_time_ms, recommendation=recommendation,
                llm_used=llm_used, sensor_used=use_sensor,
                sensor_data=sensor_data, swin_results=None,
                vote_method="single_model", majority_count="1 / 1",
            )
            saved_ok = True
        except Exception as e:
            print(f"⚠️  Gagal simpan prediksi: {e}")

    return {
        "model_key": model_key, "arsitektur": m["arch_label"],
        "dataset": m["dataset_label"],
        "source_path": m.get("path"),
        "predicted_class": predicted_class, "confidence_percentage": confidence,
        "detection_time_ms": detection_time_ms, "recommendation": recommendation,
        "prediction_id": prediction_id, "saved_to_database": saved_ok,
        "sensor_used": use_sensor, "sensor_data": sensor_data,
    }


# ═════════════════════════════════════════════════════════════════
# DETECTION — /compare
# ═════════════════════════════════════════════════════════════════
@app.post("/compare", response_model=CompareResponse, tags=["Detection"])
async def compare_all_models(file: UploadFile = File(...), use_sensor: bool = True):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")
    if not ml_models:
        raise HTTPException(status_code=503, detail="Tidak ada model yang tersedia")

    sensor_data     = _generate_sensor_data()
    sensor_statuses = _evaluate_sensor(sensor_data)
    abnormal        = [s for s in sensor_statuses if s.status != "normal"]
    sensor_info = {
        "lokasi": "Sawah Demo — Indramayu, Jawa Barat",
        "timestamp": datetime.now().isoformat(), "data": sensor_data,
        "kesimpulan": "✅ Semua kondisi sensor dalam batas normal." if not abnormal
            else f"⚠️ {len(abnormal)} parameter di luar batas: " + ", ".join(s.parameter for s in abnormal),
    }

    results: dict[str, ModelCompareResult] = {}
    for model_key, m in ml_models.items():
        try:
            t0 = time.perf_counter()
            predicted_class, confidence = predict(
                image_bytes=image_bytes, model=m["model"],
                class_names=m["class_names"], model_name=m["model_name"],
            )
            results[model_key] = ModelCompareResult(
                arsitektur=m["arch_label"], dataset=m["dataset_label"],
                predicted_class=predicted_class, confidence_percentage=confidence,
                detection_time_ms=round((time.perf_counter() - t0) * 1000, 2), status="success",
            )
        except Exception as e:
            results[model_key] = ModelCompareResult(
                arsitektur=m.get("arch_label", model_key), dataset=m.get("dataset_label", ""),
                predicted_class=None, confidence_percentage=None,
                detection_time_ms=None, status=f"error: {str(e)}",
            )

    successful = {k: v for k, v in results.items() if v.status == "success" and v.detection_time_ms}
    pred_with_conf_all = [(v.predicted_class, v.confidence_percentage or 0) for v in successful.values() if v.predicted_class]
    majority_class = _weighted_vote(pred_with_conf_all)[0] if pred_with_conf_all else None

    best_confidence_model = max(successful, key=lambda k: successful[k].confidence_percentage or 0) if successful else None
    all_times  = [v.detection_time_ms for v in successful.values()]
    arch_times : dict[str, list[float]] = {}
    ds_times   : dict[str, list[float]] = {}
    for model_key, v in successful.items():
        m = ml_models[model_key]
        arch_times.setdefault(m["arch_label"], []).append(v.detection_time_ms)
        ds_times.setdefault(m["dataset_label"], []).append(v.detection_time_ms)

    time_stats = DetectionTimeStats(
        min_ms=round(min(all_times), 2) if all_times else None,
        max_ms=round(max(all_times), 2) if all_times else None,
        avg_ms=round(sum(all_times)/len(all_times), 2) if all_times else None,
        fastest_model=min(successful, key=lambda k: successful[k].detection_time_ms) if successful else None,
        slowest_model=max(successful, key=lambda k: successful[k].detection_time_ms) if successful else None,
        stats_per_arsitektur={k: round(sum(v)/len(v), 2) for k, v in arch_times.items()},
        stats_per_dataset={k: round(sum(v)/len(v), 2) for k, v in ds_times.items()},
    )

    recommendation = None
    if majority_class:
        try:
            recommendation, _ = get_recommendation_groq_rag(majority_class, sensor_data if use_sensor else None)
        except Exception:
            pass

    return CompareResponse(
        total_models=len(ml_models), successful_models=len(successful),
        majority_class=majority_class, best_confidence_model=best_confidence_model,
        detection_time_stats=time_stats, recommendation=recommendation,
        sensor=sensor_info if use_sensor else None, results=results,
    )


# ═════════════════════════════════════════════════════════════════
# DETECTION — /compare/gabungan  (5 arsitektur × DATASET GABUNGAN)
# Membandingkan ANTAR ARSITEKTUR (Swin-B, ViT, ResNet-50, InceptionV3,
# EfficientNet-B0) yang dilatih pada dataset gabungan 14 kelas yang sama.
# ═════════════════════════════════════════════════════════════════
@app.post("/compare/gabungan", response_model=CompareResponse, tags=["Detection"])
async def compare_gabungan_models(
    file         : UploadFile    = File(...),
    use_sensor   : bool          = Form(True),
    manual_sensor: Optional[str] = Form(None),
    llm          : str           = Form("groq"),
):
    """
    Bandingkan SEMUA model yang dilatih pada DATASET GABUNGAN (14 kelas):
    Swin-B, ViT, ResNet-50, InceptionV3, EfficientNet-B0.

    Yang dibandingkan:
    - deteksi (predicted_class) tiap arsitektur
    - confidence (%)
    - kecepatan deteksi (ms) + statistik min/max/avg/tercepat/terlambat
    - rekomendasi penanganan → LLM + **RAG** untuk kelas hasil voting
    - konteks sensor (manual > ThingsBoard > dummy) yang ikut masuk prompt RAG

    Form:
    - use_sensor    : ikutkan sensor ke prompt LLM/RAG (default True)
    - manual_sensor : JSON string sensor manual (opsional, prioritas tertinggi)
    - llm           : "groq" (HIGH, default) atau "gemini" (MEDIUM) — keduanya RAG
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")
    if not ml_models_gabungan:
        raise HTTPException(
            status_code=503,
            detail=("Tidak ada model gabungan yang siap. Pastikan file .h5 ada di folder "
                    "'model_gabungan/' (swin_base_best.h5, vit_best.h5, resnet50_best.h5, "
                    "inception_v3_best.h5, efficientnet_b0_best.h5)."),
        )

    # ── Prediksi tiap arsitektur: deteksi + confidence + waktu ────
    results: dict[str, ModelCompareResult] = {}
    for model_key, m in ml_models_gabungan.items():
        try:
            t0 = time.perf_counter()
            predicted_class, confidence = predict(
                image_bytes=image_bytes, model=m["model"],
                class_names=m["class_names"], model_name=m["model_name"],
            )
            results[model_key] = ModelCompareResult(
                arsitektur=m["arch_label"], dataset=m["dataset_label"],
                predicted_class=predicted_class, confidence_percentage=confidence,
                detection_time_ms=round((time.perf_counter() - t0) * 1000, 2), status="success",
            )
        except Exception as e:
            results[model_key] = ModelCompareResult(
                arsitektur=m.get("arch_label", model_key),
                dataset=m.get("dataset_label", "Dataset Gabungan (14 kelas)"),
                predicted_class=None, confidence_percentage=None,
                detection_time_ms=None, status=f"error: {str(e)}",
            )

    successful = {k: v for k, v in results.items() if v.status == "success" and v.detection_time_ms}
    pred_with_conf_all = [(v.predicted_class, v.confidence_percentage or 0) for v in successful.values() if v.predicted_class]
    if not pred_with_conf_all:
        raise HTTPException(status_code=500, detail="Semua model gabungan gagal memproses gambar")

    # ── Voting terbobot → kelas mayoritas antar-arsitektur ───────
    majority_class        = _weighted_vote(pred_with_conf_all)[0]
    best_confidence_model = max(successful, key=lambda k: successful[k].confidence_percentage or 0)

    # ── Statistik kecepatan deteksi ─────────────────────────
    all_times  = [v.detection_time_ms for v in successful.values()]
    arch_times : dict[str, list[float]] = {}
    for model_key, v in successful.items():
        arch_times.setdefault(ml_models_gabungan[model_key]["arch_label"], []).append(v.detection_time_ms)
    time_stats = DetectionTimeStats(
        min_ms=round(min(all_times), 2), max_ms=round(max(all_times), 2),
        avg_ms=round(sum(all_times)/len(all_times), 2),
        fastest_model=min(successful, key=lambda k: successful[k].detection_time_ms),
        slowest_model=max(successful, key=lambda k: successful[k].detection_time_ms),
        stats_per_arsitektur={k: round(sum(v)/len(v), 2) for k, v in arch_times.items()},
        stats_per_dataset={"Dataset Gabungan (14 kelas)": round(sum(all_times)/len(all_times), 2)},
    )

    # ── Sensor (manual > ThingsBoard > dummy) ──────────────────
    sensor_data = _resolve_sensor_for_llm(use_sensor, manual_sensor)
    sensor_info = None
    if use_sensor and sensor_data:
        try:
            statuses = _evaluate_sensor(sensor_data)
            abnormal = [s for s in statuses if s.status != "normal"]
            sensor_info = {
                "data": sensor_data,
                "detail_status": [s.model_dump() for s in statuses],
                "kesimpulan": "✅ Semua kondisi sensor dalam batas normal." if not abnormal
                    else f"⚠️ {len(abnormal)} parameter di luar batas: " + ", ".join(s.parameter for s in abnormal),
            }
        except Exception:
            sensor_info = {"data": sensor_data}

    # ── Rekomendasi LLM + RAG untuk kelas hasil voting ──────────
    recommendation = None
    try:
        if llm == "gemini":
            recommendation, _ = get_recommendation_gemini_rag(majority_class, sensor_data if use_sensor else None)
        else:
            recommendation, _ = get_recommendation_groq_rag(majority_class, sensor_data if use_sensor else None)
    except Exception as e:
        print(f"⚠️  Rekomendasi LLM+RAG gagal: {e}")

    return CompareResponse(
        total_models=len(ml_models_gabungan), successful_models=len(successful),
        majority_class=majority_class, best_confidence_model=best_confidence_model,
        detection_time_stats=time_stats, recommendation=recommendation,
        sensor=sensor_info if use_sensor else None, results=results,
    )


# ═════════════════════════════════════════════════════════════════
# DETECTION — /compare/by-arch
# ═════════════════════════════════════════════════════════════════
@app.post("/compare/by-arch", tags=["Detection"])
async def compare_by_architecture(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    if not ml_models:
        raise HTTPException(status_code=503, detail="Tidak ada model yang tersedia")
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    grouped: dict[str, list] = {arch: [] for arch in ARCH_LABELS}
    for model_key, m in ml_models.items():
        try:
            t0 = time.perf_counter()
            predicted_class, confidence = predict(
                image_bytes=image_bytes, model=m["model"],
                class_names=m["class_names"], model_name=m["model_name"],
            )
            grouped.setdefault(m["arch_key"], []).append({
                "model_key": model_key, "dataset": m["dataset_label"],
                "predicted_class": predicted_class, "confidence_percentage": confidence,
                "detection_time_ms": round((time.perf_counter() - t0) * 1000, 2), "status": "success",
            })
        except Exception as e:
            grouped.setdefault(m["arch_key"], []).append({"model_key": model_key, "status": f"error: {str(e)}"})

    summary = {}
    for arch_key, entries in grouped.items():
        if not entries:
            continue
        ok    = [e for e in entries if e.get("status") == "success"]
        times = [e["detection_time_ms"] for e in ok]
        confs = [e["confidence_percentage"] for e in ok]
        preds = [e["predicted_class"] for e in ok]
        summary[arch_key] = {
            "label": ARCH_LABELS.get(arch_key, arch_key),
            **({"avg_confidence": round(sum(confs)/len(confs), 2),
                "avg_time_ms": round(sum(times)/len(times), 2),
                "min_time_ms": round(min(times), 2), "max_time_ms": round(max(times), 2),
                "majority_class": Counter(preds).most_common(1)[0][0]} if ok else {}),
            "results": entries,
        }
    return {"by_architecture": summary, "total_models_run": sum(len(v) for v in grouped.values())}


# ═════════════════════════════════════════════════════════════════
# LLM
# ═════════════════════════════════════════════════════════════════
@app.post("/compare-llm", tags=["LLM"])
async def compare_llm(disease_name: str = Form(...), use_sensor: bool = Form(True)):
    if not disease_name.strip():
        raise HTTPException(status_code=400, detail="disease_name tidak boleh kosong")
    sensor_data = _generate_sensor_data() if use_sensor else None
    return compare_llm_recommendation(disease_name, sensor_data)


@app.post("/compare-llm/predict", tags=["LLM"])
async def compare_llm_from_image(file: UploadFile = File(...), use_sensor: bool = True):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    swin_models = {k: v for k, v in ml_models.items() if v["arch_key"] == "swin_base"}
    if not swin_models:
        if "model" not in ml_model:
            raise HTTPException(status_code=503, detail="Tidak ada model Swin")
        swin_models = {"swin_base__default": {"model": ml_model["model"], "class_names": ml_model["class_names"],
            "model_name": ml_model["model_name"], "arch_key": "swin_base", "dataset_label": "Default"}}

    swin_results: dict = {}
    pred_list   : list = []
    for model_key, meta in swin_models.items():
        try:
            t0 = time.perf_counter()
            pc, conf = predict(image_bytes=image_bytes, model=meta["model"],
                class_names=meta["class_names"], model_name=meta["model_name"])
            swin_results[model_key] = {"dataset": meta["dataset_label"], "predicted_class": pc,
                "confidence_percentage": conf, "detection_time_ms": round((time.perf_counter()-t0)*1000,2), "status": "success"}
            pred_list.append(pc)
        except Exception as e:
            swin_results[model_key] = {"dataset": meta.get("dataset_label", model_key), "status": f"error: {str(e)}"}

    if not pred_list:
        raise HTTPException(status_code=500, detail="Semua model Swin gagal")

    pred_with_conf = [(v["predicted_class"], v["confidence_percentage"]) for v in swin_results.values() if v.get("status") == "success"]
    majority_class, weight_detail, vote_method = _weighted_vote(pred_with_conf)
    sukses = [v for v in swin_results.values() if v.get("status") == "success"]
    best_swin_key = max((k for k,v in swin_results.items() if v.get("status")=="success"), key=lambda k: swin_results[k]["confidence_percentage"], default=None)

    sensor_data = _generate_sensor_data() if use_sensor else None
    llm_result  = compare_llm_recommendation(majority_class, sensor_data)
    return {
        "swin_results": swin_results, "total_swin_models": len(swin_models),
        "successful_models": len(sukses), "majority_class": majority_class,
        "vote_detail": weight_detail, "vote_method": vote_method,
        "majority_count": f"{sum(1 for cls,_ in pred_with_conf if cls==majority_class)} / {len(pred_with_conf)} model",
        "avg_confidence": round(sum(v["confidence_percentage"] for v in sukses)/len(sukses),2),
        "avg_detection_time_ms": round(sum(v["detection_time_ms"] for v in sukses)/len(sukses),2),
        "best_swin_model": {"model_key": best_swin_key, "detail": swin_results.get(best_swin_key)},
        "sensor_data": sensor_data, "llm_comparison": llm_result,
    }


# ═════════════════════════════════════════════════════════════════
# CHATBOT
# ═════════════════════════════════════════════════════════════════
@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat(
    question       : str           = Form(...),
    disease_context: str           = Form(...),
    llm            : str           = Form("groq"),
    prediction_id  : Optional[str] = Form(None),
    use_sensor     : bool          = Form(False),
    manual_sensor  : Optional[str] = Form(None),
    x_user_id      : Optional[str] = Header(None),
):
    """
    Chatbot tanya jawab penyakit padi.
    Pesan disimpan ke tabel `chat_messages` di Supabase.
    Jika `prediction_id` diisi, chat akan terhubung ke prediksi tersebut.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong")

    # Sensor sama dengan analisa: manual (bila diisi) > ThingsBoard (historis) > none.
    # Untuk objek BUKAN padi, sensor lapangan tidak relevan -> jangan diambil sama sekali
    # agar LLM tidak menyebut/mengarang data sensor.
    _ctx = (disease_context or "").strip().lower()
    is_non_padi = _ctx in ("bukan_padi", "bukan tanaman padi") or "bukan tanaman padi" in _ctx
    sensor_data = None if is_non_padi else _resolve_sensor_for_llm(use_sensor, manual_sensor)
    if llm == "gemini":
        answer   = get_chat_response_gemini(question, disease_context, sensor_data)
        llm_used = "gemini"
    else:
        answer   = get_chat_response_groq(question, disease_context, sensor_data)
        llm_used = "groq"

    user_id = _resolve_user(x_user_id or "anonymous")
    if user_id:
        try:
            db.save_chat_message(
                user_id=user_id, question=question, answer=answer,
                disease_context=disease_context, llm_used=llm_used,
                prediction_id=prediction_id,
            )
        except Exception as e:
            print(f"⚠️  Gagal simpan chat: {e}")

    return ChatResponse(answer=answer, disease_context=disease_context, llm_used=llm_used)


# ════════════════════════════════════════��════════════════════════
# HISTORY — Predictions
# ═════════════════════════════════════════════════════════════════
@app.get("/history/{device_id}", response_model=HistoryResponse, tags=["History"])
async def get_history(device_id: str, limit: int = 20, offset: int = 0):
    """Ambil riwayat prediksi berdasarkan device_id dari Supabase."""
    try:
        user = db.get_user_by_device(device_id)
    except Exception as e:
        # DB/Supabase tidak tersedia atau env belum diisi.
        raise HTTPException(
            status_code=503,
            detail=f"Database tidak tersedia (cek SUPABASE_URL/SUPABASE_KEY): {str(e)}",
        )
    if not user:
        return HistoryResponse(
            success=True,
            history=[],
            pagination=Pagination(total=0, limit=limit, offset=offset, has_more=False),
        )

    try:
        rows, total = db.get_predictions(user["id"], limit=limit, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gagal ambil riwayat: {str(e)}")

    history_items = [
        HistoryItem(
            prediction_id=row["id"], predicted_class=row["predicted_class"],
            confidence=row["confidence"], detection_time_ms=row.get("detection_time_ms"),
            timestamp=row["created_at"], llm_used=row.get("llm_used"),
            sensor_used=row.get("sensor_used"), recommendation=row.get("recommendation"),
            vote_method=row.get("vote_method"), image_url=row.get("image_url"),
        )
        for row in rows
    ]
    return HistoryResponse(
        success=True,
        history=history_items,
        pagination=Pagination(
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total,
        ),
    )


@app.get("/history/{device_id}/detail/{prediction_id}", tags=["History"])
async def get_prediction_detail(device_id: str, prediction_id: str):
    """Detail lengkap satu prediksi (termasuk swin_results dan sensor_data)."""
    user = db.get_user_by_device(device_id)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    row = db.get_prediction_by_id(prediction_id)
    if not row or row.get("user_id") != user["id"]:
        raise HTTPException(status_code=404, detail="Prediksi tidak ditemukan")
    return row


@app.delete("/history/item/{prediction_id}", tags=["History"])
async def delete_history_item(prediction_id: str):
    """Hapus satu item riwayat dari Supabase."""
    try:
        deleted = db.delete_prediction(prediction_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gagal hapus: {str(e)}")
    return {"success": deleted, "prediction_id": prediction_id}


@app.get("/prediction/{prediction_id}/image", tags=["History"])
async def get_prediction_image(prediction_id: str):
    """URL gambar prediksi (disimpan di Supabase Storage saat /predict)."""
    row = db.get_prediction_by_id(prediction_id)
    return {"image_url": (row or {}).get("image_url")}


# ═════════════════════════════════════════════════════════════════
# HISTORY — Chat
# ════════════════════════════════════════════���════════════════════
@app.get("/chat/history/{device_id}", response_model=ChatHistoryResponse, tags=["History"])
async def get_chat_history(device_id: str, limit: int = 50):
    """Ambil riwayat chat berdasarkan device_id dari Supabase."""
    user = db.get_user_by_device(device_id)
    if not user:
        return ChatHistoryResponse(history=[], total=0)
    try:
        rows = db.get_chat_messages(user["id"], limit=limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gagal ambil chat: {str(e)}")
    items = [
        ChatHistoryItem(
            id=row["id"], question=row["question"], answer=row["answer"],
            disease_context=row.get("disease_context"), llm_used=row.get("llm_used"),
            prediction_id=row.get("prediction_id"), timestamp=row["created_at"],
        )
        for row in rows
    ]
    return ChatHistoryResponse(history=items, total=len(items))


@app.get("/chat/history/by-prediction/{prediction_id}", tags=["History"])
async def get_chat_by_prediction(prediction_id: str):
    try:
        rows = db.get_chat_by_prediction(prediction_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gagal ambil chat: {str(e)}")
    return {"prediction_id": prediction_id, "messages": rows, "total": len(rows)}


# ═════════════════════════════════════════════════════════════════
# DEBUG
# ═════════════════════════════════════════════════════════════════
@app.get("/debug/user/{device_id}", tags=["Debug"])
async def debug_user(device_id: str):
    """Cek data user dan jumlah riwayat di Supabase."""
    try:
        user = db.get_user_by_device(device_id)
        if not user:
            return {"device_id": device_id, "status": "not_found"}
        _, pred_count = db.get_predictions(user["id"], limit=1)
        chat_rows     = db.get_chat_messages(user["id"], limit=1000)
        return {
            "device_id": device_id, "user_id": user["id"],
            "total_predictions_db": user.get("total_predictions", 0),
            "prediction_count": pred_count, "chat_count": len(chat_rows),
            "first_seen": user.get("first_seen"), "last_seen": user.get("last_seen"),
            "status": "ok",
        }
    except Exception as e:
        return {"device_id": device_id, "status": "db_error", "error": str(e)}