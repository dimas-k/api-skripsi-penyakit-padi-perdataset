"""
Klien ThingsBoard (IoT) untuk integrasi data sensor Stasiun AWS.

Alur: backend login ke ThingsBoard -> dapat JWT -> ambil telemetry.
Kredensial disimpan di server (.env), aplikasi mobile cukup memanggil backend
(mobile -> backend -> ThingsBoard).

Dokumentasi REST API: https://tb.petanitech.com/swagger-ui/

Dua mode pengambilan:
  * NILAI TERAKHIR  -> endpoint latest timeseries (tanpa rentang waktu).
                       Tetap mengembalikan data terakhir walau sensor non-aktif.
  * TIME SERIES     -> dengan startTs/endTs/interval/agg (untuk grafik riwayat).
                       Contoh rentang: 1 Jan 2025 - 31 Des 2025.

Endpoint telemetry:
  GET /api/plugins/telemetry/DEVICE/{deviceId}/values/timeseries
      ?keys=wtp,whm&startTs=..&endTs=..&interval=..&limit=..&agg=AVG
"""

import os
import time
import requests
from datetime import datetime, timezone, timedelta

# ── Konfigurasi (diisi lewat .env) ───────────────────────────────
TB_BASE_URL      = os.getenv("TB_BASE_URL", "https://tb.petanitech.com").rstrip("/")
TB_USERNAME      = os.getenv("TB_USERNAME", "")
TB_PASSWORD      = os.getenv("TB_PASSWORD", "")
TB_DEVICE_NAME   = os.getenv("TB_DEVICE_NAME", "AWS-003")  # device aktif
# Default: device AWS-003 (device aktif yang memiliki data telemetry).
TB_DEVICE_ID     = os.getenv("TB_DEVICE_ID", "761f91e0-948f-11f0-9479-29de1b2ea716")
TB_TIMEOUT       = int(os.getenv("TB_TIMEOUT", "45"))
# Rentang default time series: 1 Jan - 30 Des 2025.
# Catatan: data riil hanya tersedia sekitar September-Oktober 2025.
TB_HISTORY_START = os.getenv("TB_HISTORY_START", "2025-01-01T00:00:00+07:00")
TB_HISTORY_END   = os.getenv("TB_HISTORY_END",   "2025-12-30T23:59:59+07:00")

WIB = timezone(timedelta(hours=7))


class ThingsBoardError(Exception):
    """Kesalahan saat berkomunikasi dengan ThingsBoard."""


# ══ Metadata parameter Stasiun AWS ══════════════════════════════
# code = key telemetry di ThingsBoard. min/max = ambang wajar untuk padi
# (None = parameter informatif, tanpa penilaian normal/abnormal).
AWS_PARAMS = [
    {"code": "wtp",  "label": "Suhu Udara",            "satuan": "°C",     "min": 22,  "max": 33,
     "ok": "Suhu udara mendukung pertumbuhan", "lo": "Udara terlalu dingin, pertumbuhan lambat", "hi": "Udara terlalu panas, tanaman stres"},
    {"code": "whm",  "label": "Kelembaban Udara",      "satuan": "%RH",   "min": 60,  "max": 90,
     "ok": "Kelembaban udara ideal", "lo": "Udara kering, rentan Blas", "hi": "Terlalu lembab, rentan jamur"},
    {"code": "st",   "label": "Suhu Tanah",            "satuan": "°C",     "min": 20,  "max": 30,
     "ok": "Suhu tanah ideal", "lo": "Tanah dingin, akar kurang aktif", "hi": "Tanah panas, ganggu penyerapan air"},
    {"code": "sm",   "label": "Kelembaban Tanah",      "satuan": "%",     "min": 50,  "max": 90,
     "ok": "Kelembaban tanah cukup", "lo": "Tanah kering, perlu pengairan", "hi": "Terlalu basah, rentan busuk akar"},
    {"code": "sph",  "label": "pH Tanah",              "satuan": "pH",    "min": 5.5, "max": 7.0,
     "ok": "pH optimal untuk padi", "lo": "Tanah asam, hambat serapan hara", "hi": "Tanah basa, hambat serapan Fe & Mn"},
    {"code": "sn",   "label": "Nitrogen (N)",          "satuan": "mg/kg", "min": 15,  "max": 45,
     "ok": "Nitrogen cukup", "lo": "Nitrogen rendah, daun menguning", "hi": "Nitrogen berlebih, rentan Hawar"},
    {"code": "sp",   "label": "Fosfor (P)",            "satuan": "mg/kg", "min": 8,   "max": 25,
     "ok": "Fosfor cukup", "lo": "Fosfor rendah, akar lemah", "hi": "Fosfor berlebih, ganggu serapan Zn"},
    {"code": "sk",   "label": "Kalium (K)",            "satuan": "mg/kg", "min": 100, "max": 180,
     "ok": "Kalium cukup", "lo": "Kalium rendah, rentan penyakit", "hi": "Kalium berlebih, hambat serapan Ca"},
    {"code": "ss",   "label": "Salinitas Tanah",       "satuan": "mS/cm", "min": 0,   "max": 3,
     "ok": "Salinitas aman", "lo": "Salinitas sangat rendah", "hi": "Salinitas tinggi, ganggu penyerapan air"},
    {"code": "sc",   "label": "Konduktivitas Tanah",   "satuan": "µS/cm", "min": None, "max": None, "ok": "Konduktivitas listrik tanah"},
    {"code": "stds", "label": "Total Padatan Terlarut", "satuan": "ppm",  "min": None, "max": None, "ok": "Kandungan zat terlarut tanah"},
    {"code": "elux", "label": "Intensitas Cahaya",     "satuan": "Lux",   "min": None, "max": None, "ok": "Intensitas cahaya lingkungan"},
    {"code": "wrf",  "label": "Curah Hujan",           "satuan": "mm",    "min": None, "max": None, "ok": "Curah hujan tercatat"},
    {"code": "wsr",  "label": "Radiasi Matahari",      "satuan": "W/m²",  "min": None, "max": None, "ok": "Radiasi matahari tercatat"},
    {"code": "wws",  "label": "Kecepatan Angin",       "satuan": "m/s",   "min": None, "max": None, "ok": "Kecepatan angin tercatat"},
    {"code": "wwd",  "label": "Arah Angin",            "satuan": "°",      "min": None, "max": None, "ok": "Arah angin tercatat"},
]

PARAM_BY_CODE = {p["code"]: p for p in AWS_PARAMS}
TELEMETRY_KEYS = [p["code"] for p in AWS_PARAMS]


# ══ Util waktu ═════════════════════════════════════════════
def iso_to_epoch_ms(iso: str) -> int:
    """Konversi ISO datetime (mis. '2025-01-01T00:00:00+07:00') -> epoch ms."""
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=WIB)
    return int(dt.timestamp() * 1000)


def epoch_ms_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, WIB).isoformat()


# ══ Token JWT ════════════════════════════════════════════
_token_cache = {"token": None, "exp": 0.0}


def _login() -> str:
    if not TB_USERNAME or not TB_PASSWORD:
        raise ThingsBoardError("TB_USERNAME / TB_PASSWORD belum diisi di .env")
    url = f"{TB_BASE_URL}/api/auth/login"
    try:
        resp = requests.post(
            url,
            json={"username": TB_USERNAME, "password": TB_PASSWORD},
            timeout=TB_TIMEOUT,
        )
    except requests.RequestException as e:
        raise ThingsBoardError(f"Gagal terhubung ke ThingsBoard: {e}")
    if resp.status_code != 200:
        raise ThingsBoardError(
            f"Login ThingsBoard gagal ({resp.status_code}): {resp.text[:200]}"
        )
    token = resp.json().get("token")
    if not token:
        raise ThingsBoardError("Login berhasil tetapi token kosong")
    _token_cache["token"] = token
    _token_cache["exp"] = time.time() + 3600  # refresh tiap 1 jam (TTL asli ~2.5 jam)
    return token


def _get_token() -> str:
    if _token_cache["token"] and time.time() < _token_cache["exp"]:
        return _token_cache["token"]
    return _login()


def _request(method: str, path: str, **kwargs):
    """HTTP request ke ThingsBoard dengan header auth; retry sekali bila 401."""
    url = f"{TB_BASE_URL}{path}"
    kwargs.setdefault("timeout", TB_TIMEOUT)
    for attempt in range(2):
        headers = dict(kwargs.pop("headers", {}))
        headers["X-Authorization"] = f"Bearer {_get_token()}"
        try:
            resp = requests.request(method, url, headers=headers, **kwargs)
        except requests.RequestException as e:
            raise ThingsBoardError(f"Request ke ThingsBoard gagal: {e}")
        if resp.status_code == 401 and attempt == 0:
            _token_cache["token"] = None  # paksa login ulang
            continue
        if resp.status_code != 200:
            raise ThingsBoardError(
                f"ThingsBoard {method} {path} -> {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()
    raise ThingsBoardError("Gagal setelah refresh token (401)")


# ══ Device ═══════════════════════════════════════════════
def list_devices(text_search: str = "", page_size: int = 100) -> list:
    params = {"pageSize": page_size, "page": 0}
    if text_search:
        params["textSearch"] = text_search
    data = _request("GET", "/api/tenant/devices", params=params)
    return [
        {
            "id": d.get("id", {}).get("id"),
            "name": d.get("name"),
            "type": d.get("type"),
            "label": d.get("label"),
        }
        for d in data.get("data", [])
    ]


def resolve_device_id() -> str:
    if TB_DEVICE_ID:
        return TB_DEVICE_ID
    if not TB_DEVICE_NAME:
        raise ThingsBoardError("TB_DEVICE_ID / TB_DEVICE_NAME belum diisi di .env")
    try:
        data = _request("GET", "/api/tenant/devices", params={"deviceName": TB_DEVICE_NAME})
        did = data.get("id", {}).get("id")
        if did:
            return did
    except ThingsBoardError:
        pass
    for d in list_devices(TB_DEVICE_NAME):
        if d["name"] == TB_DEVICE_NAME and d["id"]:
            return d["id"]
    raise ThingsBoardError(f"Device '{TB_DEVICE_NAME}' tidak ditemukan")


# ══ Telemetry ═══════════════════════════════════════════
def get_timeseries_keys(device_id: str) -> list:
    return _request("GET", f"/api/plugins/telemetry/DEVICE/{device_id}/keys/timeseries")


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def get_latest_values(device_id: str, keys: list = None) -> dict:
    """
    NILAI TERAKHIR tiap key (endpoint latest timeseries, tanpa rentang waktu).
    Tetap mengembalikan data terakhir walau sensor sedang non-aktif.
    Return: { key: {"ts": <ms>, "value": <float|str>}, ... }
    """
    if not keys:
        keys = TELEMETRY_KEYS
    data = _request(
        "GET",
        f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries",
        params={"keys": ",".join(keys)},
    )
    latest = {}
    for key, points in data.items():
        if points:
            latest[key] = {"ts": points[0].get("ts"), "value": _to_float(points[0].get("value"))}
    return latest


def get_timeseries(
    device_id: str,
    keys: list = None,
    start=None,
    end=None,
    interval: int = 3600000,
    agg: str = "AVG",
    limit: int = 200,
) -> dict:
    """
    Data TIME SERIES (riwayat). start/end boleh ISO string atau epoch ms.
    Default rentang: TB_HISTORY_START..TB_HISTORY_END (sepanjang 2025).
    Return: { key: [ {"ts": <ms>, "value": <str>}, ... ], ... }
    """
    if not keys:
        keys = TELEMETRY_KEYS
    if start is None:
        start = TB_HISTORY_START
    if end is None:
        end = TB_HISTORY_END
    start_ts = start if isinstance(start, int) else iso_to_epoch_ms(start)
    end_ts = end if isinstance(end, int) else iso_to_epoch_ms(end)
    params = {
        "keys": ",".join(keys),
        "startTs": start_ts,
        "endTs": end_ts,
        "limit": limit,
    }
    if interval:
        params["interval"] = interval
    if agg:
        params["agg"] = agg
    return _request(
        "GET",
        f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries",
        params=params,
    )


def latest_timestamp_iso(latest: dict):
    ts_vals = [e["ts"] for e in latest.values() if e.get("ts")]
    if not ts_vals:
        return None
    return epoch_ms_to_iso(max(ts_vals))


# Cache hasil rata-rata (data historis tak berubah -> hitung sekali saja).
_avg_cache = {}


def _is_valid_reading(value) -> bool:
    """Buang nilai sentinel/error sensor. Semua parameter AWS di sini tidak
    mungkin bernilai negatif (suhu, kelembaban, angin, NPK, pH, radiasi, hujan),
    jadi nilai < 0 (mis. -324, -326) dianggap error dan tidak diikutkan."""
    return isinstance(value, (int, float)) and value >= 0


def get_average_values(device_id: str, keys: list = None, start=None, end=None) -> dict:
    """
    RATA-RATA tiap key sepanjang PERIODE DATA (bukan nilai terakhir).
    Cocok untuk sensor non-aktif: hasilnya mewakili kondisi historis, sama
    seperti angka 'Avg' pada dashboard ThingsBoard. Nilai error/sentinel
    (negatif) dibuang sebelum dirata-rata.
    Return: { key: {"ts": <ms>, "value": <float>}, ... }  (siap untuk build_snapshot)
    """
    cache_key = (device_id, tuple(keys) if keys else None, start, end)
    if cache_key in _avg_cache:
        return _avg_cache[cache_key]
    raw = get_timeseries(
        device_id, keys=keys, start=start, end=end,
        interval=86400000, agg="AVG", limit=400,
    )
    out = {}
    for key, points in raw.items():
        vals, last_ts = [], None
        for p in points:
            v = _to_float(p.get("value"))
            if _is_valid_reading(v):
                vals.append(v)
                last_ts = p.get("ts") or last_ts
        if vals:
            out[key] = {"ts": last_ts, "value": round(sum(vals) / len(vals), 2)}
    _avg_cache[cache_key] = out
    return out


def build_snapshot(latest: dict):
    """
    Ubah nilai terakhir menjadi ringkasan siap tampil.
    Return: (data, detail, abnormal)
      data     = { code: nilai }
      detail   = [ {parameter, nilai, satuan, status, keterangan}, ... ]
      abnormal = [ nama parameter di luar batas ]
    """
    data, detail, abnormal = {}, [], []
    for p in AWS_PARAMS:
        entry = latest.get(p["code"])
        if not entry:
            continue
        val = entry.get("value")
        if not isinstance(val, (int, float)):
            continue
        val = round(float(val), 2)
        data[p["code"]] = val
        mn, mx = p.get("min"), p.get("max")
        if mn is not None and mx is not None:
            if val < mn:
                status, ket = "rendah", p.get("lo", "Di bawah batas normal")
            elif val > mx:
                status, ket = "tinggi", p.get("hi", "Di atas batas normal")
            else:
                status, ket = "normal", p.get("ok", "Dalam batas normal")
        else:
            status, ket = "info", p.get("ok", "")
        detail.append({
            "parameter": p["label"],
            "nilai": val,
            "satuan": p["satuan"],
            "status": status,
            "keterangan": ket,
        })
        if status in ("rendah", "tinggi"):
            abnormal.append(p["label"])
    return data, detail, abnormal


# == Pemetaan ke format prompt LLM ==
# Prompt LLM (llm.py) memakai key Indonesia berikut.
LLM_KEY_MAP = {
    "wtp":  "suhu_udara",
    "whm":  "kelembaban_udara",
    "st":   "suhu_tanah",
    "sm":   "kelembaban_tanah",
    "sph":  "ph_tanah",
    "sn":   "nitrogen",
    "sp":   "fosfor",
    "sk":   "kalium",
    "elux": "intensitas_cahaya",
    "wrf":  "curah_hujan",
}


def to_llm_sensor_dict(data: dict) -> dict:
    """Ubah {kode_tb: nilai} -> {key_llm: nilai} sesuai format prompt LLM."""
    out = {}
    for code, llm_key in LLM_KEY_MAP.items():
        val = data.get(code)
        if val is not None:
            out[llm_key] = val
    return out


# == Rentang ketersediaan data (untuk peringatan 'historis') ==
_coverage_cache = {}
_BULAN_ID = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def get_data_coverage(device_id: str, key: str = "wtp") -> dict:
    """
    Perkirakan rentang tanggal data tersedia (resolusi harian) agar mobile bisa
    menampilkan peringatan data historis. Ringan: agregasi per hari (<=~365 titik).
    Hasil di-cache karena data historis tidak berubah.
    """
    if device_id in _coverage_cache:
        return _coverage_cache[device_id]
    raw = get_timeseries(
        device_id, keys=[key],
        start=TB_HISTORY_START, end=TB_HISTORY_END,
        interval=86400000, agg="AVG", limit=400,
    )
    points = raw.get(key) or []
    ts_list = [p.get("ts") for p in points if p.get("ts")]
    coverage = None
    if ts_list:
        coverage = {
            "start": epoch_ms_to_iso(min(ts_list)),
            "end":   epoch_ms_to_iso(max(ts_list)),
            "days":  len(ts_list),
        }
    _coverage_cache[device_id] = coverage
    return coverage


def periode_text(coverage) -> str:
    """Format rentang jadi teks Indonesia, mis. '1 September 2025 - 30 Oktober 2025'."""
    if not coverage:
        return ""
    s = datetime.fromisoformat(coverage["start"])
    e = datetime.fromisoformat(coverage["end"])
    return f"{s.day} {_BULAN_ID[s.month]} {s.year} - {e.day} {_BULAN_ID[e.month]} {e.year}"
