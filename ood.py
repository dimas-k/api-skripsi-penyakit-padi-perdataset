"""
ood.py — Deteksi Out-of-Distribution (OOD) berbasis k-NN di ruang embedding.

Tujuan: menolak gambar yang BUKAN daun/tanaman padi (mis. mouse, wajah,
tembok) SEBELUM model klasifikasi memaksa memberi label penyakit.

Kenapa k-NN (bukan lagi Mahalanobis per-kelas)?
  - Model closed-set + softmax SELALU memaksa input apa pun ke salah satu kelas.
  - Alih-alih softmax, kita lihat FITUR (embedding pre-logits) gambar.
  - Kita simpan kumpulan embedding SEMUA gambar padi referensi (in-distribution).
    Saat inferensi, kita ukur jarak fitur gambar ke tetangga padi TERDEKAT
    (jarak ke tetangga ke-k). Foto padi -> dekat ke suatu gambar padi -> jarak
    kecil. Benda random -> jauh dari semua gambar padi -> jarak besar -> ditolak.
  - Keunggulan vs Mahalanobis per-kelas: tidak perlu tiap kelas punya banyak
    gambar untuk membentuk "pusat". Cukup ada kumpulan gambar padi yang beragam
    (label boleh tidak lengkap / flat). Cocok untuk dataset yang sebagian tak
    berlabel.

Embedding di-L2-normalize -> jarak = Euclidean pada vektor unit (setara urutan
cosine), rentangnya kecil [0, ~1.41], stabil terhadap beda pencahayaan/kontras.

Modul ini TIDAK mengubah bobot model. Ia hanya membaca file statistik
(models/ood_stats.npz) hasil hitung_statistik_ood.py. Jika file tidak ada,
gate otomatis nonaktif (fail-open) sehingga API tetap berjalan normal.
"""

import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image

# Pakai transform & device yang SAMA persis dengan pipeline prediksi,
# supaya distribusi fitur konsisten dengan saat menghitung statistik.
from model import _get_transform, device


# ═══════════════════════════════════════════════════════════
# STATE GLOBAL — diisi sekali saat startup via load_ood_stats()
# ═══════════════════════════════════════════════════════════
_OOD = {
    "loaded"   : False,
    "ref"      : None,   # np.ndarray (N, D) — embedding padi referensi (unit norm)
    "labels"   : None,   # list[str]|None — label per referensi (opsional, utk log)
    "threshold": None,   # float — ambang jarak efektif
    "base"     : None,   # float — threshold dasar dari file (sebelum scale/override)
    "k"        : 5,      # int — pakai jarak ke tetangga ke-k
    "enabled"  : os.getenv("OOD_ENABLE", "1") not in ("0", "false", "False"),
}


def _num_env(name):
    """Baca angka dari .env dgn aman: buang komentar inline (#...), spasi, kutip.
    Return float, atau None kalau kosong/ngawur."""
    raw = os.getenv(name)
    if raw is None:
        return None
    val = raw.split("#", 1)[0].strip().strip('"').strip("'")
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        print(f"⚠️  {name} di .env bukan angka valid ('{raw.strip()}') — diabaikan.")
        return None


def load_ood_stats(path: str | None = None) -> bool:
    """
    Muat statistik OOD dari file .npz. Dipanggil sekali saat startup di main.py.
    Return True jika berhasil, False jika file tidak ada / gate mati.
    """
    if not _OOD["enabled"]:
        print("ℹ️  OOD gate dinonaktifkan (OOD_ENABLE=0).")
        return False

    path = path or os.getenv("OOD_STATS_PATH", "models/ood_stats.npz")
    if not os.path.exists(path):
        print(
            f"⚠️  File statistik OOD tidak ditemukan: {path}\n"
            f"    Gate OOD NONAKTIF (fail-open) — API tetap jalan normal.\n"
            f"    Jalankan: python hitung_statistik_ood.py --train-dir <folder_gambar_padi>"
        )
        return False

    data = np.load(path, allow_pickle=True)
    ref  = data["ref_embeddings"].astype(np.float32)     # (N, D)
    # pastikan unit-norm (jaga-jaga)
    nrm = np.linalg.norm(ref, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    _OOD["ref"]    = ref / nrm
    _OOD["labels"] = list(data["ref_labels"]) if "ref_labels" in data.files else None
    _OOD["k"]      = int(data["k"]) if "k" in data.files else 5

    # Override k dari .env (OOD_K) TANPA hitung ulang npz. Menaikkan k membuat
    # skor benda non-padi yang cuma "kebetulan" dekat ke sedikit acuan ikut
    # naik, sedangkan padi asli (kluster padat) tetap rendah -> pemisah melebar.
    # Catatan: ganti k mengubah skala skor, jadi kalibrasi ulang threshold
    # (OOD_THRESHOLD / OOD_THRESHOLD_SCALE) via OOD_DEBUG.
    k_env = _num_env("OOD_K")
    if k_env is not None and int(k_env) >= 1:
        _OOD["k"] = int(k_env)

    base = float(data["threshold"])
    _OOD["base"] = base

    # ── Penyesuaian threshold TANPA hitung ulang statistik ─────────
    # Gate kelewat ketat (padi asli ikut ditolak)? Longgarkan via .env:
    #   OOD_THRESHOLD_SCALE=1.3   -> ambang x1.3 (lebih longgar)
    #   OOD_THRESHOLD=0.9         -> set ambang absolut (menimpa yang di file)
    # Skor sekarang kecil (rentang ~0..1.4), jadi angka threshold juga kecil.
    override = _num_env("OOD_THRESHOLD")
    scale    = _num_env("OOD_THRESHOLD_SCALE")
    if scale is None:
        scale = 1.0
    if override is not None:
        _OOD["threshold"] = override
        src = f"override .env (OOD_THRESHOLD={override})"
    else:
        _OOD["threshold"] = base * scale
        src = f"file × skala {scale}"

    _OOD["loaded"] = True
    N, D = _OOD["ref"].shape
    print(
        f"✅ OOD stats siap: {N} gambar referensi, dim={D}, k={_OOD['k']}, "
        f"threshold dasar={base:.4f} -> efektif={_OOD['threshold']:.4f} ({src}). "
        f"Set OOD_DEBUG=1 untuk melihat skor tiap gambar."
    )
    return True


# ═══════════════════════════════════════════════════════════
# EKSTRAKSI EMBEDDING (fitur pre-logits) — 1x forward pass
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def extract_embedding(image_bytes: bytes, model, model_name: str = "swin_base") -> np.ndarray:
    """
    Ambil vektor fitur pre-logits (embedding) dari gambar, lalu L2-normalize.
    L2-normalize membuang pengaruh magnitude (pencahayaan/kontras) supaya jarak
    stabil. WAJIB sama persis dengan normalisasi di hitung_statistik_ood.py.
    """
    transform  = _get_transform(model_name)
    img        = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    feats = model.forward_features(img_tensor)
    emb   = model.forward_head(feats, pre_logits=True)   # (1, D)
    vec   = emb.squeeze(0).float().cpu().numpy()         # (D,)
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


# ═══════════════════════════════════════════════════════════
# SKOR k-NN
# ═══════════════════════════════════════════════════════════
def knn_score(emb: np.ndarray) -> tuple[float, int]:
    """
    Jarak ke tetangga padi ke-k (k-th nearest).
    emb & ref sudah unit-norm -> cos = ref·emb, jarak Euclid = sqrt(2 - 2·cos).
    Return (skor_jarak_ke_k, index_tetangga_terdekat).
    """
    ref  = _OOD["ref"]                        # (N, D)
    k    = min(_OOD["k"], ref.shape[0])
    cos  = ref @ emb                          # (N,)
    dist = np.sqrt(np.clip(2.0 - 2.0 * cos, 0.0, None))
    idx_near = int(np.argmin(dist))
    if k <= 1:
        score = float(dist[idx_near])
    else:
        part  = np.partition(dist, k - 1)[:k]
        score = float(part.max())             # jarak ke tetangga ke-k
    return score, idx_near


def check_ood(image_bytes: bytes, model, model_name: str = "swin_base") -> dict:
    """
    Cek apakah gambar OOD (bukan padi).

    Return dict:
        {
          "active"     : bool,   # apakah gate aktif (stats termuat)
          "is_ood"     : bool,   # True jika gambar dianggap BUKAN padi
          "score"      : float,  # jarak ke tetangga padi ke-k
          "threshold"  : float,
          "nearest"    : str|None,  # label referensi terdekat (untuk logging)
        }
    Jika stats belum dimuat -> active=False, is_ood=False (fail-open).
    """
    if not (_OOD["enabled"] and _OOD["loaded"]):
        return {"active": False, "is_ood": False, "score": None,
                "threshold": None, "nearest": None}

    emb        = extract_embedding(image_bytes, model, model_name)
    score, idx = knn_score(emb)
    nearest    = _OOD["labels"][idx] if _OOD["labels"] else None
    is_ood     = score > _OOD["threshold"]

    # OOD_DEBUG=1 -> cetak skor tiap gambar (untuk kalibrasi threshold).
    # Foto padi ASLI beberapa kali, lihat skor tertingginya, lalu set
    # OOD_THRESHOLD sedikit di atas skor itu.
    if os.getenv("OOD_DEBUG", "0") not in ("0", "false", "False"):
        print(f"[OOD] skor={score:.4f}  threshold={_OOD['threshold']:.4f}  "
              f"terdekat={nearest}  -> {'BUKAN PADI' if is_ood else 'padi (lolos)'}")

    return {
        "active"   : True,
        "is_ood"   : is_ood,
        "score"    : round(score, 4),
        "threshold": round(_OOD["threshold"], 4),
        "nearest"  : nearest,
    }
