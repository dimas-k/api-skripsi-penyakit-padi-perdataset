from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# ── /users ────────────────────────────────────────────────────────
class UserResponse(BaseModel):
    """Info pengguna berdasarkan device_id (tanpa sistem login)."""
    user_id          : str
    device_id        : str
    total_predictions: int  = 0
    first_seen       : str
    last_seen        : str


# ── /predict (4 Swin + voting) ────────────────────────────────────
class SwingModelResult(BaseModel):
    dataset               : str
    predicted_class       : Optional[str]
    confidence_percentage : Optional[float]
    detection_time_ms     : Optional[float]
    status                : str


class PredictionResponse(BaseModel):
    predicted_class       : str
    confidence_percentage : float
    detection_time_ms     : float
    recommendation        : str
    prediction_id         : str
    saved_to_database     : bool = False
    swin_results          : Optional[Dict[str, SwingModelResult]] = None
    total_swin_models     : Optional[int]   = None
    successful_models     : Optional[int]   = None
    vote_detail           : Optional[Dict[str, float]] = None
    vote_method           : Optional[str]  = None
    majority_count        : Optional[str]  = None
    avg_confidence        : Optional[float] = None
    avg_detection_time_ms : Optional[float] = None
    best_swin_model       : Optional[Dict[str, Any]] = None


# ── /chat ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question       : str
    disease_context: str
    llm            : Optional[str] = "groq"


class ChatResponse(BaseModel):
    answer         : str
    disease_context: str
    llm_used       : Optional[str] = "groq"


# ── /compare-llm ─────────────────────────────────────────────────
class LLMResult(BaseModel):
    llm          : str
    answer       : Optional[str]
    response_time: Optional[float]
    status       : str


class LLMCompareResponse(BaseModel):
    disease_name: str
    results     : Dict[str, LLMResult]
    fastest_llm : Optional[str]
    sensor_used : bool


# ── /compare ─────────────────────────────────────────────────────
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
    fastest_model       : Optional[str]
    slowest_model       : Optional[str]
    stats_per_arsitektur: Optional[Dict[str, float]]
    stats_per_dataset   : Optional[Dict[str, float]]


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
    status    : str
    keterangan: str


class SensorResponse(BaseModel):
    lokasi        : str
    timestamp     : str
    data          : Dict[str, Any]
    detail_status : List[SensorStatus]
    kesimpulan    : str


# ── /history (predictions) ────────────────────────────────────────
class HistoryItem(BaseModel):
    prediction_id    : str
    predicted_class  : str
    confidence       : float
    detection_time_ms: Optional[float] = None
    timestamp        : str
    llm_used         : Optional[str]  = None
    sensor_used      : Optional[bool] = None
    recommendation   : Optional[str]  = None
    vote_method      : Optional[str]  = None


class Pagination(BaseModel):
    total : int
    limit : int
    offset: int


class HistoryResponse(BaseModel):
    history   : List[HistoryItem]
    pagination: Pagination


# ── /chat/history ────────────────────────────────────────────────
class ChatHistoryItem(BaseModel):
    id             : str
    question       : str
    answer         : str
    disease_context: Optional[str] = None
    llm_used       : Optional[str] = None
    prediction_id  : Optional[str] = None
    timestamp      : str


class ChatHistoryResponse(BaseModel):
    history: List[ChatHistoryItem]
    total  : int
