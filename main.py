import os
import time
import uuid
import random
from collections import Counter
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from schemas import (
    PredictionResponse, ChatResponse, HistoryResponse, HistoryItem,
    Pagination, SensorResponse, SensorStatus,
    CompareResponse, ModelCompareResult, DetectionTimeStats,
)
from model import load_model, load_all_models, predict, ARCH_LABELS, DATASET_LABELS
from llm import (
    get_recommendation, get_recommendation_groq, get_recommendation_gemini,
    get_chat_response_groq, get_chat_response_gemini,
    compare_llm_recommendation,
)


# ═════════════════════════════════════════════════════════════════
# IN-MEMORY STORE
# ═════════════════════════════════════════════════════════════════
history_store: dict[str, list] = {}

# ml_model  → model utama untuk /predict (Swin Base, Citra Daun Padi)
# ml_models → semua 20 model untuk /compare dan /predict/{model_key}
ml_model  = {}
ml_models = {}


# ═════════════════════════════════════════════════════════════════
# HELPER — parse model_key menjadi arch + dataset
# Key format dari load_all_models: "swin_base__Citra_Daun_Padi"
# ═════════════════════════════════════════════════════════════════
def _weighted_vote(
    predictions: list[tuple[str, float]],
    min_confidence: float      = 60.0,
    high_conf_threshold: float = 85.0,
) -> tuple[str, dict, str]:
    """
    Voting cerdas dengan empat tahap:

    Tahap 1 — Single high-confidence (>= 85%, tidak ada pesaing >= 85%):
        Langsung pakai prediksi model tersebut.
        Contoh: leaf_smut 91.6% satu-satunya >= 85% → menang.

    Tahap 2 — Multiple high-confidence (>= 85%) berbeda kelas:
        Tiebreaker HANYA antar model high-confidence itu saja.
        Model < 85% TIDAK ikut tiebreaker ini.
        Contoh: neck_blast 91.48% vs bacterial_panicle_blight 92.21%
        → bacterial_panicle_blight menang. leaf_blast 61%+64% diabaikan.

    Tahap 3 — Tidak ada yang >= 85%, weighted sum (>= 60%):
        Jumlahkan confidence per kelas, hanya model >= 60% yang ikut.

    Tahap 4 — Fallback:
        Semua model di bawah 60% (gambar buram), weighted tanpa threshold.

    Returns:
        (winner_class, weight_detail, method_used)
        weight_detail selalu berisi bobot semua kelas (untuk info di response)
    """
    if not predictions:
        return "unknown", {}, "fallback_count"

    # Hitung weight_map lengkap untuk info response (tidak dipakai sebagai penentu)
    weight_map: dict[str, float] = {}
    for cls, conf in predictions:
        weight_map[cls] = round(weight_map.get(cls, 0) + conf, 2)

    # ── Tahap 1 & 2: High-confidence models (>= 85%) ─────────────
    high_conf_preds = [
        (cls, conf) for cls, conf in predictions
        if conf >= high_conf_threshold
    ]

    if high_conf_preds:
        # Tahap 1: Hanya ada satu kelas dominan di zona high-confidence
        high_classes = set(cls for cls, _ in high_conf_preds)
        if len(high_classes) == 1:
            winner = high_conf_preds[0][0]
            return winner, weight_map, "high_confidence_override"

        # Tahap 2: Beberapa kelas bersaing di zona high-confidence
        # → tiebreaker hanya antar mereka, model rendah TIDAK ikut
        high_weight: dict[str, float] = {}
        for cls, conf in high_conf_preds:
            high_weight[cls] = round(high_weight.get(cls, 0) + conf, 2)
        winner = max(high_weight, key=lambda k: high_weight[k])
        return winner, weight_map, "high_confidence_tiebreak"

    # ── Tahap 3: Weighted sum model >= 60% ───────────────────────
    filtered = [(cls, conf) for cls, conf in predictions if conf >= min_confidence]
    if filtered:
        w: dict[str, float] = {}
        for cls, conf in filtered:
            w[cls] = round(w.get(cls, 0) + conf, 2)
        winner = max(w, key=lambda k: w[k])
        return winner, weight_map, "weighted"

    # ── Tahap 4: Fallback weighted tanpa threshold ────────────────
    winner = max(weight_map, key=lambda k: weight_map[k])
    return winner, weight_map, "weighted_no_threshold"


def _parse_model_key(model_key: str) -> tuple[str, str]:
    """
    Ekstrak arsitektur dan dataset dari model_key.
    Contoh: 'swin_base__Citra_Daun_Padi' → ('swin_base', 'Citra_Daun_Padi')
    """
    if "__" in model_key:
        arch, dataset = model_key.split("__", 1)
        return arch, dataset
    # fallback jika tidak ada separator
    for arch in ARCH_LABELS:
        if model_key.startswith(arch + "_"):
            dataset = model_key[len(arch) + 1:]
            return arch, dataset
    return model_key, "unknown"


# ═════════════════════════════════════════════════════════════════
# LIFECYCLE — startup & shutdown
# ═════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Model utama untuk /predict ─────────────────────────────
    # Dibaca dari MODEL_PATH di .env (default: Swin Base Citra Daun Padi)
    print("\n══ Loading model utama (/predict) ══")
    try:
        ml_model["model"], ml_model["class_names"], ml_model["model_name"] = \
            load_model()
        print(f"✅ Model utama siap: {ml_model['model_name']}")
    except FileNotFoundError as e:
        print(f"⚠️  Model utama tidak ditemukan: {e}")

    # ── 2. Semua 20 model untuk /compare ──────────────────────────
    # Swin (4)      → path dari .env (MODEL_swin_base_{dataset})
    # Non-Swin (16) → path hardcode di model.py
    all_loaded = load_all_models()

    for model_key, meta in all_loaded.items():
        if meta["status"] == "loaded":
            ml_models[model_key] = meta

    print(f"✅ Total model siap di ml_models: {len(ml_models)} / {len(all_loaded)}")
    print(f"   Keys: {list(ml_models.keys())}\n")

    yield

    # ── Shutdown ───────────────────────────────────────────────────
    ml_model.clear()
    ml_models.clear()


# ═════════════════════════════════════════════════════════════════
# APP
# ═════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "Paddy Disease Detection API",
    description = (
        "API deteksi penyakit daun padi — 5 arsitektur × 4 dataset = 20 model. "
        "Rekomendasi LLM: Groq (LLaMA 3.3 70B) & Gemini."
    ),
    version  = "3.1.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ═════════════════════════════════════════════════════════════════
# SENSOR — Dummy data lapangan sawah
# ═════════════════════════════════════════════════════════════════
def _generate_sensor_data() -> dict:
    """Dummy sensor data realistis untuk sawah padi Indramayu."""
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
        "suhu_udara"        : {"satuan": "°C",      "min": 20,    "max": 35,    "label": "Suhu Udara"},
        "kelembaban_udara"  : {"satuan": "%",        "min": 60,    "max": 90,    "label": "Kelembaban Udara"},
        "suhu_tanah"        : {"satuan": "°C",      "min": 20,    "max": 30,    "label": "Suhu Tanah"},
        "kelembaban_tanah"  : {"satuan": "%",        "min": 50,    "max": 80,    "label": "Kelembaban Tanah"},
        "ph_tanah"          : {"satuan": "",         "min": 5.5,   "max": 7.0,   "label": "pH Tanah"},
        "nitrogen"          : {"satuan": "mg/kg",   "min": 15,    "max": 45,    "label": "Nitrogen (N)"},
        "fosfor"            : {"satuan": "mg/kg",   "min": 8,     "max": 25,    "label": "Fosfor (P)"},
        "kalium"            : {"satuan": "mg/kg",   "min": 100,   "max": 180,   "label": "Kalium (K)"},
        "intensitas_cahaya" : {"satuan": "lux",      "min": 20000, "max": 55000, "label": "Intensitas Cahaya"},
        "curah_hujan"       : {"satuan": "mm/hari", "min": 0,     "max": 20,    "label": "Curah Hujan"},
    }
    info_map = {
        "suhu_udara"        : ("Mendukung fotosintesis optimal",      "Terlalu dingin, pertumbuhan lambat",      "Terlalu panas, stres tanaman"),
        "kelembaban_udara"  : ("Kelembaban ideal",                    "Terlalu kering, rentan Blas",             "Terlalu lembab, rentan jamur"),
        "suhu_tanah"        : ("Suhu tanah ideal",                    "Dingin, akar kurang aktif",               "Panas, ganggu penyerapan air"),
        "kelembaban_tanah"  : ("Kelembaban tanah cukup",              "Kering, perlu pengairan",                 "Terlalu basah, rentan busuk akar"),
        "ph_tanah"          : ("pH optimal untuk padi",               "Asam, hambat penyerapan unsur hara",      "Basa, hambat penyerapan Fe & Mn"),
        "nitrogen"          : ("Nitrogen cukup",                      "Nitrogen rendah, daun menguning",         "Nitrogen berlebih, rentan Hawar"),
        "fosfor"            : ("Fosfor cukup",                        "Fosfor rendah, akar lemah",               "Fosfor berlebih, ganggu penyerapan Zn"),
        "kalium"            : ("Kalium cukup",                        "Kalium rendah, rentan penyakit",          "Kalium berlebih, hambat penyerapan Ca"),
        "intensitas_cahaya" : ("Cahaya optimal",                      "Kurang cahaya, pertumbuhan lambat",       "Terlalu terik, stres panas"),
        "curah_hujan"       : ("Curah hujan ideal",                   "Kering, perlu irigasi tambahan",          "Hujan berlebih, rentan Hawar & Blas"),
    }
    statuses = []
    for key, thresh in thresholds.items():
        nilai    = data.get(key, 0)
        mn, mx   = thresh["min"], thresh["max"]
        ok, lo, hi = info_map[key]
        if nilai < mn:   status, ket = "rendah", lo
        elif nilai > mx: status, ket = "tinggi", hi
        else:            status, ket = "normal", ok
        statuses.append(SensorStatus(
            parameter  = thresh["label"],
            nilai      = nilai,
            satuan     = thresh["satuan"],
            status     = status,
            keterangan = ket,
        ))
    return statuses


# ═════════════════════════════════════════════════════════════════
# GENERAL
# ═════════════════════════════════════════════════════════════════
@app.get("/health", tags=["General"])
async def health_check():
    """Cek status API dan model yang tersedia."""
    by_arch = {}
    for key in ml_models:
        arch, _ = _parse_model_key(key)
        by_arch[arch] = by_arch.get(arch, 0) + 1
    return {
        "status"         : "healthy",
        "model_utama"    : ml_model.get("model_name", "belum_load"),
        "total_model"    : len(ml_models),
        "model_per_arch" : by_arch,
        "llm_groq"       : "llama-3.3-70b-versatile",
        "llm_gemini"     : "gemini-2.0-flash",
        "num_classes"    : len(ml_model.get("class_names", [])),
    }


@app.get("/classes", tags=["General"])
async def get_classes():
    """Daftar kelas penyakit yang dikenali model."""
    return {
        "classes": ml_model.get("class_names", []),
        "total"  : len(ml_model.get("class_names", [])),
    }


@app.get("/models", tags=["General"])
async def list_models():
    """
    Daftar semua model yang berhasil di-load.

    Key format: "{arsitektur}__{dataset}"
    - Swin  (4 model) : path dari .env
    - Lainnya (16)    : path hardcode di model.py
    """
    result = {}
    for key, meta in ml_models.items():
        result[key] = {
            "arsitektur"   : meta["arch_label"],
            "dataset"      : meta["dataset_label"],
            "model_name"   : meta["model_name"],
            "source"       : ".env" if meta["arch_key"] == "swin_base" else "hardcode",
        }
    return {
        "total_model": len(result),
        "models"     : result,
    }


@app.get("/db-test", tags=["General"])
async def db_test():
    return {"status": "ok", "message": "In-memory mode active"}


# ═════════════════════════════════════════════════════════════════
# SENSOR
# ═════════════════════════════════════════════════════════════════
@app.get("/sensor", response_model=SensorResponse, tags=["Sensor"])
async def get_sensor_data():
    """Kembalikan dummy data sensor sawah padi Indramayu."""
    data     = _generate_sensor_data()
    statuses = _evaluate_sensor(data)
    abnormal = [s for s in statuses if s.status != "normal"]
    if not abnormal:
        kesimpulan = "✅ Semua kondisi sensor dalam batas normal. Tanaman dalam kondisi baik."
    else:
        params     = ", ".join(s.parameter for s in abnormal)
        kesimpulan = (
            f"⚠️ {len(abnormal)} parameter di luar batas normal: {params}. "
            "Perlu perhatian lebih lanjut."
        )
    return SensorResponse(
        lokasi        = "Sawah Demo — Indramayu, Jawa Barat",
        timestamp     = datetime.now().isoformat(),
        data          = data,
        detail_status = statuses,
        kesimpulan    = kesimpulan,
    )


# ═════════════════════════════════════════════════════════════════
# DETECTION — /predict  (model utama: Swin Base dari MODEL_PATH)
# ═════════════════════════════════════════════════════════════════
@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict_disease(
    file         : UploadFile    = File(...),
    x_user_id    : Optional[str] = None,
    x_device_info: Optional[str] = None,
    use_sensor   : bool          = False,
    llm          : str           = "groq",
):
    """
    Upload gambar → prediksi dengan **model utama (Swin Base)** + rekomendasi LLM.

    Model utama dikonfigurasi via `MODEL_PATH` di `.env`.

    - `use_sensor=true` → sertakan data sensor dummy dalam rekomendasi LLM
    - `llm=groq`        → Groq / LLaMA 3.3 70B (default)
    - `llm=gemini`      → Gemini
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model utama belum siap")

    # ── Prediksi ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    predicted_class, confidence = predict(
        image_bytes = image_bytes,
        model       = ml_model["model"],
        class_names = ml_model["class_names"],
        model_name  = ml_model["model_name"],
    )
    detection_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    # ── Sensor & LLM ─────────────────────────────────────────────
    sensor_data = _generate_sensor_data() if use_sensor else None

    if llm == "gemini":
        recommendation = get_recommendation_gemini(predicted_class, sensor_data)
        llm_used       = "gemini"
    else:
        recommendation = get_recommendation_groq(predicted_class, sensor_data)
        llm_used       = "groq"

    # ── Simpan ke history ─────────────────────────────────────────
    prediction_id = str(uuid.uuid4())
    timestamp     = datetime.now().isoformat()
    device_id     = x_user_id or "unknown"

    history_store.setdefault(device_id, []).insert(0, {
        "prediction_id"    : prediction_id,
        "predicted_class"  : predicted_class,
        "confidence"       : confidence,
        "detection_time_ms": detection_time_ms,
        "timestamp"        : timestamp,
        "llm_used"         : llm_used,
        "sensor_used"      : use_sensor,
    })

    return PredictionResponse(
        predicted_class       = predicted_class,
        confidence_percentage = confidence,
        detection_time_ms     = detection_time_ms,
        recommendation        = recommendation,
        prediction_id         = prediction_id,
        saved_to_database     = True,
    )


# ═════════════════════════════════════════════════════════════════
# DETECTION — /predict/{model_key}  (model spesifik)
# ═════════════════════════════════════════════════════════════════
@app.post("/predict/{model_key}", tags=["Detection"])
async def predict_with_model(
    model_key : str,
    file      : UploadFile = File(...),
    use_sensor: bool       = False,
    llm       : str        = "groq",
):
    """
    Upload gambar → prediksi dengan model spesifik (arsitektur + dataset).

    **Format model_key**: `{arsitektur}__{dataset}`

    Contoh:
    - `swin_base__Citra_Daun_Padi`
    - `vit__JENIS_PENYAKIT_PADI`
    - `resnet50__Paddy_disease`

    Lihat daftar lengkap di `GET /models`.
    """
    if model_key not in ml_models:
        available = list(ml_models.keys())
        raise HTTPException(
            status_code = 404,
            detail      = f"Model '{model_key}' tidak tersedia. "
                          f"Lihat GET /models. Tersedia: {available}",
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    m = ml_models[model_key]

    # ── Prediksi ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    predicted_class, confidence = predict(
        image_bytes = image_bytes,
        model       = m["model"],
        class_names = m["class_names"],
        model_name  = m["model_name"],
    )
    detection_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    # ── Sensor & LLM ─────────────────────────────────────────────
    sensor_data = _generate_sensor_data() if use_sensor else None

    if llm == "gemini":
        recommendation = get_recommendation_gemini(predicted_class, sensor_data)
    else:
        recommendation = get_recommendation_groq(predicted_class, sensor_data)

    return {
        "model_key"            : model_key,
        "arsitektur"           : m["arch_label"],
        "dataset"              : m["dataset_label"],
        "source_path"          : ".env" if m["arch_key"] == "swin_base" else "hardcode",
        "predicted_class"      : predicted_class,
        "confidence_percentage": confidence,
        "detection_time_ms"    : detection_time_ms,
        "recommendation"       : recommendation,
        "sensor_used"          : use_sensor,
        "sensor_data"          : sensor_data,
    }


# ═════════════════════════════════════════════════════════════════
# DETECTION — /compare  (semua 20 model + statistik kecepatan)
# ═════════════════════════════════════════════════════════════════
@app.post("/compare", response_model=CompareResponse, tags=["Detection"])
async def compare_all_models(
    file      : UploadFile = File(...),
    use_sensor: bool       = True,
):
    """
    Upload 1 gambar → jalankan **semua 20 model** (5 arsitektur × 4 dataset).

    Setiap model menghasilkan:
    - `predicted_class` + `confidence_percentage`
    - `detection_time_ms` — waktu inferensi model tersebut

    Ringkasan mencakup:
    - `majority_class`        — kelas paling banyak diprediksi (voting)
    - `best_confidence_model` — model dengan confidence tertinggi
    - `detection_time_stats`  — min/max/avg + rata-rata per arsitektur & dataset
    - `recommendation`        — rekomendasi LLM untuk kelas mayoritas
    - `sensor`                — data sensor dummy sawah (opsional via `use_sensor`)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    if not ml_models:
        raise HTTPException(status_code=503, detail="Tidak ada model yang tersedia")

    # ── Sensor dummy ──────────────────────────────────────────────
    sensor_data     = _generate_sensor_data()
    sensor_statuses = _evaluate_sensor(sensor_data)
    abnormal        = [s for s in sensor_statuses if s.status != "normal"]
    sensor_info = {
        "lokasi"    : "Sawah Demo — Indramayu, Jawa Barat",
        "timestamp" : datetime.now().isoformat(),
        "data"      : sensor_data,
        "kesimpulan": (
            "✅ Semua kondisi sensor dalam batas normal."
            if not abnormal else
            f"⚠️ {len(abnormal)} parameter di luar batas: "
            + ", ".join(s.parameter for s in abnormal)
        ),
    }

    # ── Prediksi semua model (sequential) ────────────────────────
    results: dict[str, ModelCompareResult] = {}

    for model_key, m in ml_models.items():
        try:
            t0 = time.perf_counter()
            predicted_class, confidence = predict(
                image_bytes = image_bytes,
                model       = m["model"],
                class_names = m["class_names"],
                model_name  = m["model_name"],
            )
            detection_time_ms = round((time.perf_counter() - t0) * 1000, 2)

            results[model_key] = ModelCompareResult(
                arsitektur            = m["arch_label"],
                dataset               = m["dataset_label"],
                predicted_class       = predicted_class,
                confidence_percentage = confidence,
                detection_time_ms     = detection_time_ms,
                status                = "success",
            )

        except Exception as e:
            results[model_key] = ModelCompareResult(
                arsitektur            = m.get("arch_label", model_key),
                dataset               = m.get("dataset_label", ""),
                predicted_class       = None,
                confidence_percentage = None,
                detection_time_ms     = None,
                status                = f"error: {str(e)}",
            )

    # ── Pisahkan yang sukses ──────────────────────────────────────
    successful = {
        k: v for k, v in results.items()
        if v.status == "success" and v.detection_time_ms is not None
    }

    # ── Majority voting ───────────────────────────────────────────
    # Confidence-weighted voting — model confidence < 60% diabaikan
    pred_with_conf_all = [
        (v.predicted_class, v.confidence_percentage or 0)
        for v in successful.values() if v.predicted_class
    ]
    if pred_with_conf_all:
        majority_class, _, _ = _weighted_vote(pred_with_conf_all, min_confidence=60.0)
    else:
        majority_class = None

    # ── Model terbaik berdasarkan confidence ─────────────────────
    best_confidence_model = (
        max(successful, key=lambda k: successful[k].confidence_percentage or 0)
        if successful else None
    )

    # ── Statistik waktu deteksi ───────────────────────────────────
    all_times  = [v.detection_time_ms for v in successful.values()]
    arch_times : dict[str, list[float]] = {}
    ds_times   : dict[str, list[float]] = {}

    for model_key, v in successful.items():
        m          = ml_models[model_key]
        arch_label = m["arch_label"]
        ds_label   = m["dataset_label"]
        arch_times.setdefault(arch_label, []).append(v.detection_time_ms)
        ds_times.setdefault(ds_label, []).append(v.detection_time_ms)

    stats_per_arsitektur = {
        k: round(sum(v) / len(v), 2) for k, v in arch_times.items()
    }
    stats_per_dataset = {
        k: round(sum(v) / len(v), 2) for k, v in ds_times.items()
    }

    fastest_model = (
        min(successful, key=lambda k: successful[k].detection_time_ms)
        if successful else None
    )
    slowest_model = (
        max(successful, key=lambda k: successful[k].detection_time_ms)
        if successful else None
    )

    time_stats = DetectionTimeStats(
        min_ms               = round(min(all_times), 2)               if all_times else None,
        max_ms               = round(max(all_times), 2)               if all_times else None,
        avg_ms               = round(sum(all_times) / len(all_times), 2) if all_times else None,
        fastest_model        = fastest_model,
        slowest_model        = slowest_model,
        stats_per_arsitektur = stats_per_arsitektur,
        stats_per_dataset    = stats_per_dataset,
    )

    # ── Rekomendasi LLM untuk kelas mayoritas ─────────────────────
    recommendation = None
    if majority_class:
        try:
            sd             = sensor_data if use_sensor else None
            recommendation = get_recommendation_groq(majority_class, sd)
        except Exception:
            recommendation = None

    return CompareResponse(
        total_models          = len(ml_models),
        successful_models     = len(successful),
        majority_class        = majority_class,
        best_confidence_model = best_confidence_model,
        detection_time_stats  = time_stats,
        recommendation        = recommendation,
        sensor                = sensor_info if use_sensor else None,
        results               = results,
    )


# ═════════════════════════════════════════════════════════════════
# DETECTION — /compare/by-arch  (dikelompokkan per arsitektur)
# ═════════════════════════════════════════════════════════════════
@app.post("/compare/by-arch", tags=["Detection"])
async def compare_by_architecture(file: UploadFile = File(...)):
    """
    Upload gambar → jalankan semua 20 model, hasil dikelompokkan per arsitektur.

    Setiap arsitektur menampilkan:
    - Hasil 4 dataset
    - `avg_confidence`, `avg_time_ms`, `min_time_ms`, `max_time_ms`
    - `majority_class` untuk arsitektur tersebut
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    if not ml_models:
        raise HTTPException(status_code=503, detail="Tidak ada model yang tersedia")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    grouped: dict[str, list] = {arch: [] for arch in ARCH_LABELS}

    for model_key, m in ml_models.items():
        arch_key = m["arch_key"]
        try:
            t0 = time.perf_counter()
            predicted_class, confidence = predict(
                image_bytes = image_bytes,
                model       = m["model"],
                class_names = m["class_names"],
                model_name  = m["model_name"],
            )
            detection_time_ms = round((time.perf_counter() - t0) * 1000, 2)

            grouped.setdefault(arch_key, []).append({
                "model_key"            : model_key,
                "dataset"              : m["dataset_label"],
                "predicted_class"      : predicted_class,
                "confidence_percentage": confidence,
                "detection_time_ms"    : detection_time_ms,
                "status"               : "success",
            })
        except Exception as e:
            grouped.setdefault(arch_key, []).append({
                "model_key": model_key,
                "dataset"  : m.get("dataset_label", ""),
                "status"   : f"error: {str(e)}",
            })

    summary = {}
    for arch_key, entries in grouped.items():
        if not entries:
            continue
        successful = [e for e in entries if e.get("status") == "success"]
        if not successful:
            summary[arch_key] = {
                "label"  : ARCH_LABELS.get(arch_key, arch_key),
                "results": entries,
            }
            continue

        times = [e["detection_time_ms"] for e in successful]
        confs = [e["confidence_percentage"] for e in successful]
        preds = [e["predicted_class"] for e in successful]

        summary[arch_key] = {
            "label"          : ARCH_LABELS.get(arch_key, arch_key),
            "avg_confidence" : round(sum(confs) / len(confs), 2),
            "avg_time_ms"    : round(sum(times) / len(times), 2),
            "min_time_ms"    : round(min(times), 2),
            "max_time_ms"    : round(max(times), 2),
            "majority_class" : Counter(preds).most_common(1)[0][0],
            "results"        : entries,
        }

    return {
        "by_architecture" : summary,
        "total_models_run": sum(len(v) for v in grouped.values()),
    }


# ═════════════════════════════════════════════════════════════════
# LLM — Groq vs Gemini
# ═════════════════════════════════════════════════════════════════
@app.post("/compare-llm", tags=["LLM"])
async def compare_llm(
    disease_name: str  = Form(...),
    use_sensor  : bool = Form(True),
):
    """
    Bandingkan rekomendasi Groq (LLaMA 3.3 70B) vs Gemini.
    Sensor dummy disertakan secara default.
    """
    if not disease_name.strip():
        raise HTTPException(status_code=400, detail="disease_name tidak boleh kosong")

    sensor_data = _generate_sensor_data() if use_sensor else None
    result      = compare_llm_recommendation(disease_name, sensor_data)
    return result


@app.post("/compare-llm/predict", tags=["LLM"])
async def compare_llm_from_image(
    file      : UploadFile = File(...),
    use_sensor: bool       = True,
):
    """
    Upload gambar → deteksi dengan **semua 4 Swin Transformer** (tiap dataset)
    → majority voting → bandingkan rekomendasi Groq vs Gemini.

    Flow:
    1. Gambar diproses oleh 4 Swin (Citra Daun Padi, Jenis Penyakit Padi,
       Paddy V3 Augmentasi, Paddy Disease Classification)
    2. Hasil 4 prediksi di-voting → kelas mayoritas
    3. Kelas mayoritas dikirim ke Groq & Gemini untuk rekomendasi
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    # ── Ambil semua 4 Swin dari ml_models ────────────────────────
    swin_models = {
        key: meta for key, meta in ml_models.items()
        if meta["arch_key"] == "swin_base"
    }

    # Fallback ke model utama jika Swin di ml_models belum terisi
    if not swin_models:
        if "model" not in ml_model:
            raise HTTPException(
                status_code = 503,
                detail      = "Tidak ada model Swin yang tersedia",
            )
        swin_models = {
            "swin_base__default": {
                "model"        : ml_model["model"],
                "class_names"  : ml_model["class_names"],
                "model_name"   : ml_model["model_name"],
                "arch_key"     : "swin_base",
                "dataset_label": "Default (MODEL_PATH)",
            }
        }

    # ── Prediksi ke-4 Swin ────────────────────────────────────────
    swin_results    : dict = {}
    all_predictions : list = []

    for model_key, meta in swin_models.items():
        try:
            t0 = time.perf_counter()
            predicted_class, confidence = predict(
                image_bytes = image_bytes,
                model       = meta["model"],
                class_names = meta["class_names"],
                model_name  = meta["model_name"],
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            swin_results[model_key] = {
                "dataset"              : meta["dataset_label"],
                "predicted_class"      : predicted_class,
                "confidence_percentage": confidence,
                "detection_time_ms"    : elapsed_ms,
                "status"               : "success",
            }
            all_predictions.append(predicted_class)

        except Exception as e:
            swin_results[model_key] = {
                "dataset": meta.get("dataset_label", model_key),
                "status" : f"error: {str(e)}",
            }

    if not all_predictions:
        raise HTTPException(
            status_code = 500,
            detail      = "Semua model Swin gagal melakukan prediksi",
        )

    # ── Confidence-weighted voting dari 4 Swin ────────────────────
    # Kumpulkan (kelas, confidence) hanya dari model yang sukses.
    # Model dengan confidence < 60% diabaikan dari voting agar tidak
    # "meracuni" hasil — misalnya dua model ragu (41%, 57%) mengalahkan
    # satu model yang sangat yakin (92%).
    pred_with_conf = [
        (v["predicted_class"], v["confidence_percentage"])
        for v in swin_results.values()
        if v.get("status") == "success"
    ]
    majority_class, weight_detail, vote_method = _weighted_vote(
        pred_with_conf, min_confidence=60.0
    )
    majority_count = sum(1 for cls, _ in pred_with_conf if cls == majority_class)

    # Swin dengan confidence tertinggi
    best_swin_key = max(
        (k for k, v in swin_results.items() if v.get("status") == "success"),
        key = lambda k: swin_results[k]["confidence_percentage"],
        default = None,
    )
    best_swin = swin_results[best_swin_key] if best_swin_key else None

    # Rata-rata confidence & waktu dari yang sukses
    sukses        = [v for v in swin_results.values() if v.get("status") == "success"]
    avg_confidence = round(sum(v["confidence_percentage"] for v in sukses) / len(sukses), 2)
    avg_time_ms    = round(sum(v["detection_time_ms"] for v in sukses) / len(sukses), 2)

    # ── Sensor & LLM comparison ───────────────────────────────────
    sensor_data = _generate_sensor_data() if use_sensor else None
    llm_result  = compare_llm_recommendation(majority_class, sensor_data)

    return {
        # ── Hasil tiap Swin ───────────────────────────────────────
        "swin_results"         : swin_results,
        "total_swin_models"    : len(swin_models),
        "successful_models"    : len(sukses),

        # ── Voting ────────────────────────────────────────────────
        "majority_class"       : majority_class,
        "vote_detail"          : weight_detail,
        "vote_method"          : vote_method,
        "majority_count"       : f"{majority_count} / {len(pred_with_conf)} model pilih kelas ini",

        # ── Statistik Swin ────────────────────────────────────────
        "avg_confidence"       : avg_confidence,
        "avg_detection_time_ms": avg_time_ms,
        "best_swin_model"      : {
            "model_key": best_swin_key,
            "detail"   : best_swin,
        },

        # ── Sensor & LLM ─────────────────────────────────────────
        "sensor_data"          : sensor_data,
        "llm_comparison"       : llm_result,
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
    x_user_id      : Optional[str] = None,
):
    """Chatbot tanya jawab penyakit padi. `llm`: `groq` (default) atau `gemini`."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong")

    if llm == "gemini":
        answer   = get_chat_response_gemini(question, disease_context)
        llm_used = "gemini"
    else:
        answer   = get_chat_response_groq(question, disease_context)
        llm_used = "groq"

    return ChatResponse(answer=answer, disease_context=disease_context, llm_used=llm_used)


# ═════════════════════════════════════════════════════════════════
# HISTORY
# ═════════════════════════════════════════════════════════════════
@app.get("/history/{device_id}", response_model=HistoryResponse, tags=["History"])
async def get_history(device_id: str, limit: int = 20, offset: int = 0):
    """Ambil riwayat prediksi berdasarkan device_id."""
    items     = history_store.get(device_id, [])
    paginated = items[offset: offset + limit]
    return HistoryResponse(
        history    = [HistoryItem(**item) for item in paginated],
        pagination = Pagination(total=len(items), limit=limit, offset=offset),
    )


@app.delete("/history/item/{prediction_id}", tags=["History"])
async def delete_history_item(prediction_id: str):
    """Hapus satu item riwayat berdasarkan prediction_id."""
    deleted = False
    for device_id, items in history_store.items():
        before = len(items)
        history_store[device_id] = [
            i for i in items if i["prediction_id"] != prediction_id
        ]
        if len(history_store[device_id]) < before:
            deleted = True
            break
    return {"success": deleted, "prediction_id": prediction_id}


@app.get("/prediction/{prediction_id}/image", tags=["History"])
async def get_prediction_image(prediction_id: str):
    return {"image_url": None}


# ═════════════════════════════════════════════════════════════════
# DEBUG
# ═════════════════════════════════════════════════════════════════
@app.get("/debug/user/{device_id}", tags=["Debug"])
async def debug_user(device_id: str):
    return {
        "device_id"    : device_id,
        "history_count": len(history_store.get(device_id, [])),
        "status"       : "ok",
    }
