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
from model import load_model, load_model_from_path, predict
from llm import (
    get_recommendation, get_recommendation_groq, get_recommendation_gemini,
    get_chat_response_groq, get_chat_response_gemini,
    compare_llm_recommendation,
)


# ── In-memory stores ──────────────────────────────────────────────
history_store: dict[str, list] = {}

ml_model  = {}   # model utama (/predict)
ml_models = {}   # semua 20 model (/compare, /predict/{model_key})

# ── 20 Model Files: 5 arsitektur × 4 dataset ─────────────────────
MODEL_FILES = {
    # ── EfficientNet-B0 ──────────────────────────────────────────
    "efficientnet_b0_citra_daun_padi"       : "models/efficientnet_b0_Citra_Daun_Padi_best.h5",
    "efficientnet_b0_jenis_penyakit_padi"   : "models/efficientnet_b0_JENIS_PENYAKIT_PADI_best.h5",
    "efficientnet_b0_paddy_v3_augmentasi"   : "models/efficientnet_b0_paddy-dataset-v3-augmentasi_best.h5",
    "efficientnet_b0_paddy_disease_classif" : "models/efficientnet_b0_Paddy-disease-classification_best.h5",

    # ── InceptionV3 ──────────────────────────────────────────────
    "inception_v3_citra_daun_padi"          : "models/inception_v3_Citra_Daun_Padi_best.h5",
    "inception_v3_jenis_penyakit_padi"      : "models/inception_v3_JENIS_PENYAKIT_PADI_best.h5",
    "inception_v3_paddy_v3_augmentasi"      : "models/inception_v3_paddy-dataset-v3-augmentasi_best.h5",
    "inception_v3_paddy_disease_classif"    : "models/inception_v3_Paddy-disease-classification_best.h5",

    # ── ResNet-50 ─────────────────────────────────────────────────
    "resnet50_citra_daun_padi"              : "models/resnet50_Citra_Daun_Padi_best.h5",
    "resnet50_jenis_penyakit_padi"          : "models/resnet50_JENIS_PENYAKIT_PADI_best.h5",
    "resnet50_paddy_v3_augmentasi"          : "models/resnet50_paddy-dataset-v3-augmentasi_best.h5",
    "resnet50_paddy_disease_classif"        : "models/resnet50_Paddy-disease-classification_best.h5",

    # ── Swin Transformer Base ─────────────────────────────────────
    "swin_base_citra_daun_padi"             : "models/swin_base_Citra_Daun_Padi_best.h5",
    "swin_base_jenis_penyakit_padi"         : "models/swin_base_JENIS_PENYAKIT_PADI_best.h5",
    "swin_base_paddy_v3_augmentasi"         : "models/swin_base_paddy-dataset-v3-augmentasi_best.h5",
    "swin_base_paddy_disease_classif"       : "models/swin_base_Paddy-disease-classification_best.h5",

    # ── ViT-Base/16 ───────────────────────────────────────────────
    "vit_citra_daun_padi"                   : "models/vit_Citra_Daun_Padi_best.h5",
    "vit_jenis_penyakit_padi"               : "models/vit_JENIS_PENYAKIT_PADI_best.h5",
    "vit_paddy_v3_augmentasi"               : "models/vit_paddy-dataset-v3-augmentasi_best.h5",
    "vit_paddy_disease_classif"             : "models/vit_Paddy-disease-classification_best.h5",
}

DATASET_LABELS = {
    "citra_daun_padi"       : "Citra Daun Padi",
    "jenis_penyakit_padi"   : "Jenis Penyakit Padi",
    "paddy_v3_augmentasi"   : "Paddy Dataset V3 Augmentasi",
    "paddy_disease_classif" : "Paddy Disease Classification",
}

ARCH_LABELS = {
    "efficientnet_b0" : "EfficientNet-B0",
    "inception_v3"    : "InceptionV3",
    "resnet50"        : "ResNet-50",
    "swin_base"       : "Swin Transformer Base",
    "vit"             : "ViT-Base/16",
}


def _parse_model_key(model_key: str) -> tuple[str, str]:
    """
    Ekstrak arsitektur dan dataset dari model_key.
    Contoh: 'swin_base_citra_daun_padi' → ('swin_base', 'citra_daun_padi')
    """
    for arch in ARCH_LABELS:
        if model_key.startswith(arch + "_"):
            dataset_key = model_key[len(arch) + 1:]
            return arch, dataset_key
    return model_key, "unknown"


# ── Lifecycle ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model utama untuk /predict — Swin Base, dataset Citra Daun Padi
    default_key  = "swin_base_citra_daun_padi"
    default_path = MODEL_FILES[default_key]

    if os.path.exists(default_path):
        ml_model["model"], ml_model["class_names"], ml_model["model_name"] = \
            load_model_from_path(default_path)
        ml_model["dataset_key"] = "citra_daun_padi"
    else:
        try:
            ml_model["model"], ml_model["class_names"], ml_model["model_name"] = \
                load_model()
            ml_model["dataset_key"] = "unknown"
        except FileNotFoundError as e:
            print(f"⚠️  Model utama tidak ditemukan: {e}")

    # Load semua 20 model untuk /compare
    print("\n── Loading semua model (5 arsitektur × 4 dataset) ──")
    for key, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                model, class_names, model_name = load_model_from_path(path)
                arch_key, dataset_key = _parse_model_key(key)
                ml_models[key] = {
                    "model"        : model,
                    "class_names"  : class_names,
                    "model_name"   : model_name,
                    "arch_key"     : arch_key,
                    "dataset_key"  : dataset_key,
                    "arch_label"   : ARCH_LABELS.get(arch_key, arch_key),
                    "dataset_label": DATASET_LABELS.get(dataset_key, dataset_key),
                }
            except Exception as e:
                print(f"⚠️  Gagal load {key}: {e}")
        else:
            print(f"⚠️  {path} tidak ditemukan, skip")

    print(f"\n✅ Total model tersedia: {len(ml_models)} / {len(MODEL_FILES)}")
    print(f"   {list(ml_models.keys())}\n")

    yield
    ml_model.clear()
    ml_models.clear()


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Paddy Disease Detection API",
    description = (
        "API deteksi penyakit daun padi — 5 arsitektur × 4 dataset = 20 model. "
        "Rekomendasi LLM: Groq (LLaMA 3.3 70B) & Gemini 2.5 Flash."
    ),
    version  = "3.0.0",
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
        "ph_tanah"          : round(random.uniform(5.5, 7.2),  1),
        "nitrogen"          : round(random.uniform(10.0, 50.0), 1),
        "fosfor"            : round(random.uniform(5.0,  30.0), 1),
        "kalium"            : round(random.uniform(80.0, 200.0), 1),
        "intensitas_cahaya" : round(random.uniform(15000, 60000), 0),
        "curah_hujan"       : round(random.uniform(0.0,  25.0), 1),
    }


def _evaluate_sensor(data: dict) -> list[SensorStatus]:
    thresholds = {
        "suhu_udara"       : {"satuan": "°C",      "min": 20,    "max": 35,    "label": "Suhu Udara"},
        "kelembaban_udara" : {"satuan": "%",        "min": 60,    "max": 90,    "label": "Kelembaban Udara"},
        "suhu_tanah"       : {"satuan": "°C",      "min": 20,    "max": 30,    "label": "Suhu Tanah"},
        "kelembaban_tanah" : {"satuan": "%",        "min": 50,    "max": 80,    "label": "Kelembaban Tanah"},
        "ph_tanah"         : {"satuan": "",         "min": 5.5,   "max": 7.0,   "label": "pH Tanah"},
        "nitrogen"         : {"satuan": "mg/kg",   "min": 15,    "max": 45,    "label": "Nitrogen (N)"},
        "fosfor"           : {"satuan": "mg/kg",   "min": 8,     "max": 25,    "label": "Fosfor (P)"},
        "kalium"           : {"satuan": "mg/kg",   "min": 100,   "max": 180,   "label": "Kalium (K)"},
        "intensitas_cahaya": {"satuan": "lux",      "min": 20000, "max": 55000, "label": "Intensitas Cahaya"},
        "curah_hujan"      : {"satuan": "mm/hari", "min": 0,     "max": 20,    "label": "Curah Hujan"},
    }
    info_map = {
        "suhu_udara"        : ("Mendukung fotosintesis optimal",       "Terlalu dingin, pertumbuhan lambat",      "Terlalu panas, stres tanaman"),
        "kelembaban_udara"  : ("Kelembaban ideal",                     "Terlalu kering, rentan Blas",              "Terlalu lembab, rentan jamur"),
        "suhu_tanah"        : ("Suhu tanah ideal",                     "Dingin, akar kurang aktif",               "Panas, ganggu penyerapan air"),
        "kelembaban_tanah"  : ("Kelembaban tanah cukup",               "Kering, perlu pengairan",                 "Terlalu basah, rentan busuk akar"),
        "ph_tanah"          : ("pH optimal untuk padi",                "Asam, hambat penyerapan unsur hara",      "Basa, hambat penyerapan Fe & Mn"),
        "nitrogen"          : ("Nitrogen cukup",                       "Nitrogen rendah, daun menguning",         "Nitrogen berlebih, rentan Hawar"),
        "fosfor"            : ("Fosfor cukup",                         "Fosfor rendah, akar lemah",               "Fosfor berlebih, ganggu penyerapan Zn"),
        "kalium"            : ("Kalium cukup",                         "Kalium rendah, rentan penyakit",          "Kalium berlebih, hambat penyerapan Ca"),
        "intensitas_cahaya" : ("Cahaya optimal",                       "Kurang cahaya, pertumbuhan lambat",       "Terlalu terik, stres panas"),
        "curah_hujan"       : ("Curah hujan ideal",                    "Kering, perlu irigasi tambahan",          "Hujan berlebih, rentan Hawar & Blas"),
    }
    statuses = []
    for key, thresh in thresholds.items():
        nilai = data.get(key, 0)
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
            f"Perlu perhatian lebih lanjut."
        )
    return SensorResponse(
        lokasi        = "Sawah Demo — Indramayu, Jawa Barat",
        timestamp     = datetime.now().isoformat(),
        data          = data,
        detail_status = statuses,
        kesimpulan    = kesimpulan,
    )


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
        "status"         : "healthy",
        "model_utama"    : ml_model.get("model_name", "unknown"),
        "total_model"    : len(ml_models),
        "model_per_arch" : by_arch,
        "llm_groq"       : "llama-3.3-70b-versatile",
        "llm_gemini"     : "gemini-2.5-flash",
        "num_classes"    : len(ml_model.get("class_names", [])),
    }


@app.get("/classes", tags=["General"])
async def get_classes():
    return {
        "classes": ml_model.get("class_names", []),
        "total"  : len(ml_model.get("class_names", [])),
    }


@app.get("/models", tags=["General"])
async def list_models():
    """Daftar semua model yang tersedia (arsitektur × dataset)."""
    result = {}
    for key, data in ml_models.items():
        result[key] = {
            "arsitektur": data["arch_label"],
            "dataset"   : data["dataset_label"],
            "model_name": data["model_name"],
        }
    return {
        "total_model": len(result),
        "models"     : result,
    }


@app.get("/db-test", tags=["General"])
async def db_test():
    return {"status": "ok", "message": "In-memory mode active"}


# ═════════════════════════════════════════════════════════════════
# DETECTION — /predict (model utama: Swin Base, Citra Daun Padi)
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
    Upload gambar → prediksi dengan **Swin Base (Citra Daun Padi)** + rekomendasi LLM.

    - `use_sensor=true` → sertakan data sensor dummy dalam rekomendasi
    - `llm=groq`        → Groq / LLaMA 3.3 70B (default)
    - `llm=gemini`      → Gemini 2.5 Flash
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model utama belum siap")

    t0 = time.perf_counter()
    predicted_class, confidence = predict(
        image_bytes = image_bytes,
        model       = ml_model["model"],
        class_names = ml_model["class_names"],
        model_name  = ml_model["model_name"],
    )
    detection_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    sensor_data = _generate_sensor_data() if use_sensor else None

    if llm == "gemini":
        recommendation = get_recommendation_gemini(predicted_class, sensor_data)
        llm_used       = "gemini"
    else:
        recommendation = get_recommendation_groq(predicted_class, sensor_data)
        llm_used       = "groq"

    prediction_id = str(uuid.uuid4())
    timestamp     = datetime.now().isoformat()

    device_id = x_user_id or "unknown"
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
# DETECTION — /predict/{model_key} (model tertentu)
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

    Contoh model_key: `swin_base_citra_daun_padi`, `vit_jenis_penyakit_padi`

    Lihat daftar lengkap di GET /models.
    """
    if model_key not in ml_models:
        raise HTTPException(
            status_code = 404,
            detail      = f"Model '{model_key}' tidak tersedia. Lihat GET /models.",
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB")

    m = ml_models[model_key]

    t0 = time.perf_counter()
    predicted_class, confidence = predict(
        image_bytes = image_bytes,
        model       = m["model"],
        class_names = m["class_names"],
        model_name  = m["model_name"],
    )
    detection_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    sensor_data = _generate_sensor_data() if use_sensor else None

    if llm == "gemini":
        recommendation = get_recommendation_gemini(predicted_class, sensor_data)
    else:
        recommendation = get_recommendation_groq(predicted_class, sensor_data)

    return {
        "model_key"            : model_key,
        "arsitektur"           : m["arch_label"],
        "dataset"              : m["dataset_label"],
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
    - `majority_class`          — kelas paling banyak diprediksi (voting)
    - `best_confidence_model`   — model dengan confidence tertinggi
    - `detection_time_stats`    — min/max/avg global + rata-rata per arsitektur & per dataset
    - `recommendation`          — rekomendasi LLM untuk kelas mayoritas
    - `sensor`                  — data sensor dummy sawah (opsional via `use_sensor`)
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
    all_preds   = [v.predicted_class for v in successful.values() if v.predicted_class]
    majority_class = Counter(all_preds).most_common(1)[0][0] if all_preds else None

    # ── Model terbaik berdasarkan confidence ─────────────────────
    best_confidence_model = (
        max(successful, key=lambda k: successful[k].confidence_percentage or 0)
        if successful else None
    )

    # ── Statistik waktu deteksi ───────────────────────────────────
    all_times = [v.detection_time_ms for v in successful.values()]

    # Rata-rata per arsitektur (pakai arch_key internal, tampilkan label)
    arch_times: dict[str, list[float]] = {}
    ds_times  : dict[str, list[float]] = {}

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
        min_ms               = round(min(all_times), 2) if all_times else None,
        max_ms               = round(max(all_times), 2) if all_times else None,
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
            sd = sensor_data if use_sensor else None
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
# DETECTION — /compare/by-arch  (bandingkan per arsitektur)
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
    Bandingkan rekomendasi Groq (LLaMA 3.3 70B) vs Gemini 2.5 Flash.
    Sensor dummy **selalu disertakan** secara default.
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
    Upload gambar → deteksi (Swin Base) → bandingkan rekomendasi Groq vs Gemini.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model utama belum siap")

    image_bytes = await file.read()

    t0 = time.perf_counter()
    predicted_class, confidence = predict(
        image_bytes = image_bytes,
        model       = ml_model["model"],
        class_names = ml_model["class_names"],
        model_name  = ml_model["model_name"],
    )
    detection_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    sensor_data = _generate_sensor_data() if use_sensor else None
    llm_result  = compare_llm_recommendation(predicted_class, sensor_data)

    return {
        "predicted_class"      : predicted_class,
        "confidence_percentage": confidence,
        "detection_time_ms"    : detection_time_ms,
        "model_used"           : ml_model["model_name"],
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
    items     = history_store.get(device_id, [])
    paginated = items[offset: offset + limit]
    return HistoryResponse(
        history    = [HistoryItem(**item) for item in paginated],
        pagination = Pagination(total=len(items), limit=limit, offset=offset),
    )


@app.delete("/history/item/{prediction_id}", tags=["History"])
async def delete_history_item(prediction_id: str):
    deleted = False
    for device_id, items in history_store.items():
        before = len(items)
        history_store[device_id] = [i for i in items if i["prediction_id"] != prediction_id]
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
