from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# ── /predict ─────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class       : str
    confidence_percentage : float
    detection_time_ms     : float
    recommendation        : str
    prediction_id         : str
    saved_to_database     : bool = False


# ── /chat ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question       : str
    disease_context: str
    llm            : Optional[str] = "groq"   # "groq" | "gemini"


class ChatResponse(BaseModel):
    answer         : str
    disease_context: str
    llm_used       : Optional[str] = "groq"


# ── /compare-llm ─────────────────────────────────────────────────
class LLMResult(BaseModel):
    llm          : str
    answer       : Optional[str]
    response_time: Optional[float]   # detik
    status       : str


class LLMCompareResponse(BaseModel):
    disease_name: str
    results     : Dict[str, LLMResult]
    fastest_llm : Optional[str]
    sensor_used : bool


# ── /compare (model vision) ───────────────────────────────────────
class ModelCompareResult(BaseModel):
    arsitektur            : str
    dataset               : str
    predicted_class       : Optional[str]
    confidence_percentage : Optional[float]
    detection_time_ms     : Optional[float]
    status                : str


class DetectionTimeStats(BaseModel):
    min_ms              : Optional[float]
    max_ms              : Optional[float]
    avg_ms              : Optional[float]
    fastest_model       : Optional[str]    # model_key tercepat
    slowest_model       : Optional[str]    # model_key terlambat
    stats_per_arsitektur: Optional[Dict[str, float]]   # rata-rata ms per arsitektur
    stats_per_dataset   : Optional[Dict[str, float]]   # rata-rata ms per dataset


class CompareResponse(BaseModel):
    total_models          : int
    successful_models     : int
    majority_class        : Optional[str]
    best_confidence_model : Optional[str]
    detection_time_stats  : DetectionTimeStats
    recommendation        : Optional[str]
    sensor                : Optional[Dict[str, Any]]
    results               : Dict[str, ModelCompareResult]


# ── /sensor ──────────────────────────────────────────────────────
class SensorStatus(BaseModel):
    parameter : str
    nilai     : float
    satuan    : str
    status    : str        # "normal" | "rendah" | "tinggi"
    keterangan: str


class SensorResponse(BaseModel):
    lokasi        : str
    timestamp     : str
    data          : Dict[str, Any]
    detail_status : List[SensorStatus]
    kesimpulan    : str


# ── /history ─────────────────────────────────────────────────────
class HistoryItem(BaseModel):
    prediction_id    : str
    predicted_class  : str
    confidence       : float
    detection_time_ms: Optional[float] = None
    timestamp        : str


class Pagination(BaseModel):
    total : int
    limit : int
    offset: int


class HistoryResponse(BaseModel):
    history   : List[HistoryItem]
    pagination: Pagination
