"""
hitung_statistik_ood.py — Bangun statistik OOD berbasis k-NN.

DIJALANKAN SEKALI DI LOKAL (butuh gambar padi + torch/timm terpasang).
Output: models/ood_stats.npz -> dipakai otomatis oleh API (ood.py).

Langkah:
  1. Muat model GABUNGAN utama (sama dgn /predict, dari MODEL_PATH di .env).
  2. Kumpulkan SEMUA gambar padi dari folder --train-dir (boleh beberapa folder,
     boleh berstruktur kelas ATAU flat/tak berlabel — label TIDAK dipakai untuk
     skor, hanya untuk info tetangga terdekat).
  3. Ekstrak embedding pre-logits tiap gambar, lalu L2-normalize.
  4. Simpan matriks embedding referensi (N x D) sebagai "acuan padi".
  5. Kalibrasi THRESHOLD: untuk tiap gambar --val-dir (atau leave-one-out pada
     referensi bila --val-dir kosong), hitung jarak ke tetangga padi ke-k,
     lalu threshold = persentil ke-P (default 99) x --scale.
  6. Simpan ref_embeddings, ref_labels, threshold, k ke models/ood_stats.npz.

Contoh:
  python hitung_statistik_ood.py \\
    --train-dir "train model" "paddy-v3/train" "paddy-disease-classification/test_images" \\
    --val-dir "paddy-v3/val"
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import load_model, _get_transform, device


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}


class FlatImages(Dataset):
    """
    Kumpulkan SEMUA file gambar di bawah folder-folder yang diberikan (rekursif),
    TANPA peduli struktur kelas. Jadi folder berstruktur kelas (ImageFolder)
    maupun folder flat (semua gambar langsung di dalamnya) sama-sama didukung.
    label = nama folder induk terdekat (hanya untuk info/logging).
    """
    def __init__(self, dirs, transform):
        if isinstance(dirs, str):
            dirs = [dirs]
        self.transform = transform
        self.paths, self.labels = [], []
        for d in dirs:
            for root, _, files in os.walk(d):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in IMG_EXT:
                        self.paths.append(os.path.join(root, fn))
                        self.labels.append(os.path.basename(root))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), i


@torch.no_grad()
def extract_all(model, loader, model_name):
    """Ekstrak embedding pre-logits + L2-normalize untuk semua gambar. Return (N, D)."""
    feats = []
    total = len(loader.dataset)
    done  = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        f    = model.forward_features(imgs)
        emb  = model.forward_head(f, pre_logits=True)   # (B, D)
        e    = emb.float().cpu().numpy()
        e    = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)  # L2-normalize
        feats.append(e.astype(np.float32))
        done += imgs.size(0)
        print(f"   ...embedding {done}/{total}", end="\r")
    print()
    return np.concatenate(feats, axis=0)


def knn_kth_dist(queries, ref, k, exclude_self=False, chunk=512):
    """
    Untuk tiap baris queries, jarak Euclid ke tetangga ke-k di ref
    (embedding unit-norm -> dist = sqrt(2 - 2·cos)). Diproses per-chunk agar hemat
    memori. exclude_self=True: buang jarak ~0 ke diri sendiri (leave-one-out).
    """
    out = np.empty(queries.shape[0], dtype=np.float64)
    kk  = (k + 1) if exclude_self else k
    kk  = min(kk, ref.shape[0])
    col = (k if exclude_self else k - 1)
    col = min(col, kk - 1)
    for s in range(0, queries.shape[0], chunk):
        q    = queries[s:s + chunk]
        cos  = q @ ref.T                                   # (c, N)
        dist = np.sqrt(np.clip(2.0 - 2.0 * cos, 0.0, None))
        part = np.partition(dist, kk - 1, axis=1)[:, :kk]
        part.sort(axis=1)
        out[s:s + chunk] = part[:, col]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, nargs="+",
                    help="Satu atau beberapa folder gambar padi REFERENSI "
                         "(boleh berstruktur kelas atau flat/tak berlabel). "
                         "Sebutkan semua sumber agar keragaman padi lengkap.")
    ap.add_argument("--val-dir", default=None, nargs="+",
                    help="(DISARANKAN) Folder gambar padi VALIDASI utk kalibrasi "
                         "threshold. Kosong -> leave-one-out pada referensi.")
    ap.add_argument("--out", default="models/ood_stats.npz",
                    help="Path output (default: models/ood_stats.npz)")
    ap.add_argument("--k", type=int, default=5,
                    help="Tetangga ke-k untuk skor jarak (default 5). "
                         "Lebih besar = lebih tahan noise, sedikit lebih ketat.")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="Persentil jarak padi utk threshold (default 99). "
                         "Makin tinggi = makin longgar (jarang menolak padi asli).")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Faktor pelonggar threshold akhir (default 1.0). "
                         "Mis. 1.2-1.5 kalau padi asli masih ketolak.")
    ap.add_argument("--max-refs", type=int, default=20000,
                    help="Batas maksimum embedding referensi (subsample acak bila "
                         "lebih), agar file & inferensi tetap ringan. Default 20000.")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    # ── 1. Muat model utama (gabungan) ────────────────────────
    print("══ Muat model utama (gabungan) ══")
    model, class_names, model_name = load_model()
    model.eval()
    transform = _get_transform(model_name)

    # ── 2. Kumpulkan gambar padi referensi (in-distribution) ──────
    print("══ Kumpulkan gambar padi referensi ══")
    ref_ds = FlatImages(args.train_dir, transform)
    print(f"   Folder : {args.train_dir}")
    print(f"   Total  : {len(ref_ds)} gambar padi referensi")
    if len(ref_ds) == 0:
        raise SystemExit("❌ Tidak ada gambar ditemukan di --train-dir. Cek path-nya.")
    ref_loader = DataLoader(ref_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    print("══ Ekstrak embedding referensi (pre-logits, L2-normalized) ══")
    R      = extract_all(model, ref_loader, model_name)      # (N, D) float32
    labels = np.array(ref_ds.labels, dtype=object)

    # subsample bila terlalu banyak (jaga ukuran file & kecepatan inferensi)
    if R.shape[0] > args.max_refs:
        idx = np.random.RandomState(42).choice(R.shape[0], args.max_refs, replace=False)
        R, labels = R[idx], labels[idx]
        print(f"   Subsample -> {R.shape[0]} referensi (batas --max-refs).")
    N, D = R.shape

    # ── 3. Kalibrasi threshold (jarak ke tetangga ke-k) ──────────
    print("══ Kalibrasi threshold (jarak ke tetangga ke-k) ══")
    Rd = R.astype(np.float64)
    if args.val_dir:
        val_ds = FlatImages(args.val_dir, transform)
        print(f"   Val    : {len(val_ds)} gambar dari {args.val_dir}")
        if len(val_ds) == 0:
            raise SystemExit("❌ Tidak ada gambar di --val-dir. Cek path-nya.")
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
        Xv      = extract_all(model, val_loader, model_name).astype(np.float64)
        id_dist = knn_kth_dist(Xv, Rd, args.k, exclude_self=False)
        sumber  = f"validasi ({len(val_ds)} gambar)"
    else:
        print("   ⚠️  Tanpa --val-dir: leave-one-out pada referensi.")
        id_dist = knn_kth_dist(Rd, Rd, args.k, exclude_self=True)
        sumber  = f"referensi LOO ({N} gambar)"

    base      = float(np.percentile(id_dist, args.percentile))
    threshold = base * args.scale
    print(f"   Sumber    : {sumber}")
    print(f"   Jarak ID  : min={id_dist.min():.4f} median={np.median(id_dist):.4f} "
          f"p{args.percentile:.0f}={base:.4f} max={id_dist.max():.4f}")
    print(f"   Threshold : p{args.percentile:.0f} × skala {args.scale} = {threshold:.4f}")
    print(f"   (skor rentang ~0..1.41; padi asli < threshold, benda random > threshold)")

    # ── 4. Simpan ──────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        ref_embeddings=R.astype(np.float32),
        ref_labels=labels,
        threshold=np.float32(threshold),
        k=np.int64(args.k),
        percentile=np.float32(args.percentile),
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"✅ Tersimpan: {args.out} ({size_mb:.1f} MB, {N} referensi, dim {D}, k={args.k})")
    print("   Taruh file ini di folder models/ (lokal & produksi), lalu restart uvicorn.")
    print("   Kalibrasi akhir: OOD_DEBUG=1 + foto HP asli (padi vs benda random),")
    print("   lalu atur OOD_THRESHOLD / OOD_THRESHOLD_SCALE di .env bila perlu.")


if __name__ == "__main__":
    main()
