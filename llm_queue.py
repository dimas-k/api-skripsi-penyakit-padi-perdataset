"""
llm_queue.py — Antrian (Queue) & Rate Limiter untuk pemanggilan LLM
====================================================================
Tujuan: mencegah error rate-limit (HTTP 429 "Too Many Requests") dari
penyedia LLM cloud (Gemini & Groq) yang membatasi jumlah request per
menit (RPM). Ini menjawab permintaan: "buat antrian pemanggilan LLM
sehingga tidak kena limit request per minute".

Tiga lapis proteksi:
  1. QUEUE / ANTRIAN     — semua panggilan LLM di-serialkan lewat sebuah
     semaphore. Tidak ada ledakan request bersamaan; kalau slot penuh,
     request berikutnya MENGANTRE (menunggu giliran).
  2. RATE LIMIT (RPM)    — tiap provider punya jatah maksimum request per
     60 detik (sliding window). Kalau jatah dalam 1 menit sudah habis,
     request berikutnya otomatis MENUNGGU sampai ada slot kosong.
  3. RETRY + BACKOFF     — kalau tetap kena 429, otomatis dicoba ulang
     dengan jeda yang membesar (exponential backoff).

Semua mekanisme thread-safe (aman untuk banyak request paralel).

Konfigurasi lewat .env (opsional, ada nilai default aman):
  LLM_MAX_CONCURRENT   = 1     # berapa panggilan LLM boleh jalan bersamaan
  LLM_GEMINI_RPM       = 10    # jatah request/menit Gemini (free tier ~10-15)
  LLM_GROQ_RPM         = 25    # jatah request/menit Groq   (free tier ~30)
  LLM_OLLAMA_RPM       = 1000  # lokal, praktis tanpa batas
  LLM_MAX_RETRIES      = 3     # maksimum retry saat kena 429
  LLM_RETRY_BASE_DELAY = 2.0   # jeda awal retry (detik), naik 2x tiap gagal
"""

import os
import time
import logging
import threading
import functools
from collections import deque

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# HELPER BACA ENV
# ═════════════════════════════════════════════════════════════════
def _int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# ═════════════════════════════════════════════════════════════════
# KONFIGURASI
# ═════════════════════════════════════════════════════════════════
MAX_CONCURRENT = max(1, _int_env("LLM_MAX_CONCURRENT", 1))

PROVIDER_RPM = {
    "gemini": _int_env("LLM_GEMINI_RPM", 10),
    "groq":   _int_env("LLM_GROQ_RPM", 25),
    "ollama": _int_env("LLM_OLLAMA_RPM", 1000),
}

MAX_RETRIES      = _int_env("LLM_MAX_RETRIES", 3)
RETRY_BASE_DELAY = _float_env("LLM_RETRY_BASE_DELAY", 2.0)


# ═════════════════════════════════════════════════════════════════
# SLIDING-WINDOW RATE LIMITER
# ═════════════════════════════════════════════════════════════════
class SlidingWindowRateLimiter:
    """Membatasi maksimum `max_calls` panggilan dalam `period` detik.

    Memakai sliding window: menyimpan timestamp tiap panggilan, membuang
    yang sudah lewat dari jendela, lalu memblokir (menunggu) bila jatah
    dalam jendela sudah penuh.
    """

    def __init__(self, max_calls: int, period: float = 60.0, name: str = ""):
        self.max_calls = max(1, int(max_calls))
        self.period    = float(period)
        self.name      = name
        self._calls: deque[float] = deque()
        self._lock     = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                # buang timestamp yang sudah keluar dari jendela
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                # jatah penuh — hitung lama menunggu sampai slot tertua kedaluwarsa
                wait = self.period - (now - self._calls[0]) + 0.01

            logger.info(
                f"[LLM-QUEUE] Jatah '{self.name}' penuh "
                f"({self.max_calls}/{self.period:.0f}s). Mengantre {wait:.1f}s..."
            )
            time.sleep(max(0.05, wait))


# ═════════════════════════════════════════════════════════════════
# OBJEK GLOBAL
# ═════════════════════════════════════════════════════════════════
_concurrency = threading.BoundedSemaphore(MAX_CONCURRENT)
_limiters = {
    p: SlidingWindowRateLimiter(rpm, 60.0, p) for p, rpm in PROVIDER_RPM.items()
}
_default_limiter = SlidingWindowRateLimiter(60, 60.0, "default")


def _get_limiter(provider: str) -> SlidingWindowRateLimiter:
    return _limiters.get(provider, _default_limiter)


print("✅ LLM Queue aktif:")
print(f"   Max paralel : {MAX_CONCURRENT} panggilan")
print(f"   Jatah/menit : gemini={PROVIDER_RPM['gemini']} | "
      f"groq={PROVIDER_RPM['groq']} | ollama={PROVIDER_RPM['ollama']}")
print(f"   Retry 429   : {MAX_RETRIES}x (backoff mulai {RETRY_BASE_DELAY:.1f}s)")


# ═════════════════════════════════════════════════════════════════
# CONTEXT MANAGER: masuk antrian + patuhi rate limit
# ═════════════════════════════════════════════════════════════════
class _Slot:
    def __init__(self, provider: str):
        self.provider = provider

    def __enter__(self):
        _concurrency.acquire()
        try:
            _get_limiter(self.provider).acquire()
        except BaseException:
            _concurrency.release()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        _concurrency.release()
        return False


def slot(provider: str) -> "_Slot":
    """Context manager. Contoh:
        with llm_queue.slot("gemini"):
            ... panggil API Gemini ...
    """
    return _Slot(provider)


# ═════════════════════════════════════════════════════════════════
# DECORATOR: bungkus fungsi pemanggil LLM
# ═════════════════════════════════════════════════════════════════
def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    needles = ("429", "too many requests", "rate limit", "rate-limit",
               "quota", "resource_exhausted", "resource exhausted")
    return any(n in msg for n in needles)


def rate_limited(provider: str, max_retries: int | None = None,
                 base_delay: float | None = None):
    """Decorator: setiap panggilan fungsi akan (1) mengantre lewat semaphore,
    (2) menunggu bila jatah RPM habis, dan (3) retry otomatis bila kena 429.
    """
    _max_retries = MAX_RETRIES if max_retries is None else max_retries
    _base_delay  = RETRY_BASE_DELAY if base_delay is None else base_delay

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                delay = 0.0
                with slot(provider):
                    try:
                        return fn(*args, **kwargs)
                    except Exception as e:  # noqa: BLE001
                        if _is_rate_limit_error(e) and attempt < _max_retries:
                            attempt += 1
                            delay = _base_delay * (2 ** (attempt - 1))
                            logger.warning(
                                f"[LLM-QUEUE] '{provider}' kena rate-limit (429). "
                                f"Retry {attempt}/{_max_retries} dalam {delay:.1f}s..."
                            )
                        else:
                            raise
                # tidur DI LUAR slot supaya provider lain tetap jalan saat backoff
                time.sleep(delay)

        return wrapper

    return decorator
