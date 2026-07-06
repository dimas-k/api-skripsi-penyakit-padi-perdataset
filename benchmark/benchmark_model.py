import os
import sys
import time
import argparse

import numpy as np

# ===========================================================================
# ===== KONFIGURASI =========================================================
# ===========================================================================

BATCH_SIZE = 32

GABUNGAN = {
    "name": "Gabungan (swin_base)",
    "path": "model_gabungan/swin_base_best.h5",
}

BENCHMARK_DATASETS = [
    {
        "name": "paddy-v3-augmentasi",
        "test_dir": r"D:\coolyeah\coolyeah-kok-gini-amat\apa_ini_tetiba_sem_8_aja\skripsi_ketar_ketir\skripsi-fix\dataset\dataset_paddy_v3\paddy-dataset-v3-augmentasi\test",
        "satuan_path": "models_perdataset/swin_base_paddy-dataset-v3-augmentasi_best.h5",
        # Uji LINTAS-DATASET: model satuan dari dataset LAIN diuji di test set
        # paddy_v3 (generalisasi). Urutan kelas tiap model diambil dari checkpoint.
        "extra_models": [
            {"tipe": "Satuan-Citra_Daun_Padi (lintas)",
             "path": "models_perdataset/swin_base_Citra_Daun_Padi_best.h5"},
            {"tipe": "Satuan-JENIS_PENYAKIT_PADI (lintas)",
             "path": "models_perdataset/swin_base_JENIS_PENYAKIT_PADI_best.h5"},
            {"tipe": "Satuan-Paddy_disease (lintas)",
             "path": "models_perdataset/swin_base_Paddy-disease-classification_best.h5"},
        ],
    },
    # Dataset lain DIKELUARKAN dari benchmark akurasi:
    #   - Citra_Daun_Padi        : tidak punya test set
    #   - (dataset ke-4)         : tidak punya test set
    #   - JENIS_PENYAKIT_PADI /  : test set FLAT tanpa folder per-kelas
    #     paddy-disease-classif.   (tak ada label -> akurasi tak bisa dihitung)
    # Jadi hanya paddy_v3 yang punya test set berlabel (folder per kelas).
    # Kalau nanti ada dataset lain dgn folder per-kelas, tambahkan format sama.
]

OUT_XLSX = "hasil/hasil_benchmark.xlsx"
OUT_CSV = "hasil/hasil_benchmark.csv"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Alias nama kelas -> kanonik (untuk folder yang namanya tak standar).
ALIASES = {
    "normal": "healthy",
    "blast": "leaf_blast",   # dataset paddy-v3 pakai 'Blast (Blas)' utk leaf_blast
}

# ===========================================================================
# ===== IMPOR DARI API (model.py) ATAU FALLBACK =============================
# ===========================================================================

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HAVE_API = False
try:
    from model import CLASS_NAMES as _API_CLASSES, _get_transform as _api_transform
    HAVE_API = True
except Exception as e:  # pragma: no cover
    print(f"[WARN] Tidak bisa impor dari model.py ({e}).")
    print("       Pakai definisi bawaan (swin_base + normalisasi ImageNet).")

_FALLBACK_CLASSES = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight",
    "brown_spot", "dead_heart", "downy_mildew", "healthy", "hispa",
    "leaf_blast", "leaf_smut", "neck_blast", "sheath_blight", "tungro",
    "harvest_stage",
]

CLASS_NAMES = list(_API_CLASSES) if HAVE_API else _FALLBACK_CLASSES


def build_model(num_classes):
    """Bangun swin_base via timm dgn num_classes sesuai checkpoint."""
    import timm
    return timm.create_model(
        "swin_base_patch4_window7_224", pretrained=False,
        num_classes=num_classes, drop_rate=0.3, drop_path_rate=0.2,
    )


def get_transform():
    if HAVE_API:
        return _api_transform("swin_base")
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ===========================================================================
# ===== PENCOCOKAN NAMA KELAS ===============================================
# ===========================================================================


def norm_label(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# (key_norm, canonical) dari CLASS_NAMES + ALIASES, urut terpanjang dulu.
_MATCHER = sorted(
    [(norm_label(c), c) for c in CLASS_NAMES]
    + [(norm_label(a), t) for a, t in ALIASES.items()],
    key=lambda kv: len(kv[0]), reverse=True,
)


def to_canonical(label):
    """Cocokkan nama folder bebas (mis. 'Blast (Blas)') -> nama kanonik."""
    n = norm_label(label)
    for k, c in _MATCHER:      # exact
        if k == n:
            return c
    for k, c in _MATCHER:      # prefix (folder diawali nama kelas kanonik)
        if n.startswith(k):
            return c
    for k, c in _MATCHER:      # substring
        if k in n:
            return c
    return str(label)


# ===========================================================================
# ===== LOAD CHECKPOINT =====================================================
# ===========================================================================


def _extract_state(path):
    """Return (state_dict, classes_or_None, wrapper_keys_or_None)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    classes, wrapper_keys, state = None, None, ckpt
    if isinstance(ckpt, dict):
        picked = None
        for k in ("model_state_dict", "state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                picked = ckpt[k]
                break
        if picked is not None:
            state = picked
            wrapper_keys = list(ckpt.keys())
            # cari info urutan kelas kalau tersimpan
            for ck in ("classes", "class_names"):
                if isinstance(ckpt.get(ck), (list, tuple)):
                    classes = list(ckpt[ck])
                    break
            if classes is None and isinstance(ckpt.get("idx_to_class"), dict):
                d = ckpt["idx_to_class"]
                classes = [d[i] for i in sorted(d)]
            if classes is None and isinstance(ckpt.get("class_to_idx"), dict):
                d = ckpt["class_to_idx"]
                classes = [c for c, _ in sorted(d.items(), key=lambda kv: kv[1])]
        else:
            state = ckpt  # ckpt itu sendiri state_dict

    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    return state, classes, wrapper_keys


def _infer_num_classes(state):
    for k in ("head.fc.weight", "head.weight", "fc.weight", "classifier.weight"):
        if k in state:
            return int(state[k].shape[0])
    return None


def load_model(path):
    """Return (model, num_classes, ckpt_classes, wrapper_keys)."""
    state, ckpt_classes, wrapper_keys = _extract_state(path)
    ncls = _infer_num_classes(state) or len(CLASS_NAMES)
    model = build_model(ncls)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"    [warn] load_state_dict: missing={len(missing)} "
              f"unexpected={len(unexpected)}")
    model.to(DEVICE).eval()
    return model, ncls, ckpt_classes, wrapper_keys


# ===========================================================================
# ===== DATA & PREDIKSI =====================================================
# ===========================================================================


def get_dataset_classes(test_dir):
    """Nama subfolder kelas, urut spt PyTorch ImageFolder (case-sensitive)."""
    return sorted([d for d in os.listdir(test_dir)
                   if os.path.isdir(os.path.join(test_dir, d))])


def list_test_images(test_dir, limit=None):
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"test_dir tidak ditemukan: {test_dir}")
    samples = []
    for cls in sorted(os.listdir(test_dir)):
        cdir = os.path.join(test_dir, cls)
        if not os.path.isdir(cdir):
            continue
        canon = to_canonical(cls)
        n = 0
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(IMG_EXTS):
                samples.append((os.path.join(cdir, fn), canon))
                n += 1
                if limit and n >= limit:
                    break
    return samples


@torch.no_grad()
def predict_dir(model, samples, class_list):
    """Return (list_pred_canonical, total_infer_time_detik)."""
    from PIL import Image
    transform = get_transform()
    preds = []
    total_t = 0.0
    buf = []

    def flush():
        nonlocal total_t
        if not buf:
            return
        x = torch.stack(buf).to(DEVICE)
        t0 = time.perf_counter()
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        total_t += time.perf_counter() - t0
        idx = out.argmax(dim=1).cpu().numpy()
        for j in idx:
            j = int(j)
            raw = class_list[j] if j < len(class_list) else str(j)
            preds.append(to_canonical(raw))
        buf.clear()

    for p, _ in samples:
        img = Image.open(p).convert("RGB")
        buf.append(transform(img))
        if len(buf) >= BATCH_SIZE:
            flush()
    flush()
    return preds, total_t


def compute_metrics(y_true, y_pred, labels):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, classification_report)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return {"accuracy": acc, "precision_macro": prec,
            "recall_macro": rec, "f1_macro": f1, "report": report}


def file_size_mb(path):
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    except OSError:
        return float("nan")


def eval_model(model_path, samples, fallback_classes, dataset_name, tipe):
    model, ncls, ckpt_classes, wrapper_keys = load_model(model_path)

    if ckpt_classes and len(ckpt_classes) == ncls:
        class_list = [to_canonical(c) for c in ckpt_classes]
        print(f"    [info] {ncls} kelas dari checkpoint: {list(ckpt_classes)}")
        print(f"           -> kanonik: {class_list}")
    else:
        class_list = fallback_classes
        if wrapper_keys and not ckpt_classes:
            print(f"    [info] checkpoint keys: {wrapper_keys}")
        if len(class_list) != ncls:
            print(f"    [warn] jumlah kelas checkpoint ({ncls}) != daftar "
                  f"({len(class_list)}); mapping output bisa meleset")

    preds, t = predict_dir(model, samples, class_list)
    y_true = [lbl for _, lbl in samples]
    labels = sorted(set(y_true))
    m = compute_metrics(y_true, preds, labels)
    n = len(samples)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {
        "dataset": dataset_name,
        "tipe": tipe,
        "model": os.path.basename(model_path),
        "n_kelas_model": ncls,
        "n_test": n,
        "accuracy": round(m["accuracy"], 4),
        "precision_macro": round(m["precision_macro"], 4),
        "recall_macro": round(m["recall_macro"], 4),
        "f1_macro": round(m["f1_macro"], 4),
        "waktu_per_gambar_ms": round((t / n) * 1000, 2) if n else 0,
        "ukuran_mb": file_size_mb(model_path),
        "_report": m["report"],
    }


def save_results(rows):
    import pandas as pd
    cols = ["dataset", "tipe", "model", "n_kelas_model", "n_test", "accuracy",
            "precision_macro", "recall_macro", "f1_macro",
            "waktu_per_gambar_ms", "ukuran_mb"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    df.to_csv(OUT_CSV, index=False)
    try:
        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="benchmark", index=False)
    except Exception as e:
        print(f"  [WARN] gagal tulis xlsx: {e} (CSV tetap tersimpan)")
    print("\n================ RINGKASAN BENCHMARK ================")
    print(df.to_string(index=False))
    for r in rows:
        print(f"\n----- Classification report: {r['tipe']} @ {r['dataset']} -----")
        print(r["_report"])


def main():
    ap = argparse.ArgumentParser(description="Benchmark gabungan vs satuan (PyTorch)")
    ap.add_argument("--limit", type=int, default=None,
                    help="maks gambar per kelas (untuk cek cepat)")
    args = ap.parse_args()

    print(f"Device: {DEVICE} | Sumber transform: "
          f"{'model.py API' if HAVE_API else 'fallback'} | kelas gabungan: {len(CLASS_NAMES)}")

    rows = []
    for ds in BENCHMARK_DATASETS:
        print(f"\n########## DATASET: {ds['name']} ##########")
        try:
            samples = list_test_images(ds["test_dir"], limit=args.limit)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue
        print(f"  Total gambar test: {len(samples)}")
        if not samples:
            print("  [SKIP] tidak ada gambar (cek struktur folder per-kelas).")
            continue

        dataset_classes = get_dataset_classes(ds["test_dir"])
        print(f"  Kelas dataset ({len(dataset_classes)}): {dataset_classes}")

        try:
            print("  -> evaluasi Model GABUNGAN")
            rows.append(eval_model(GABUNGAN["path"], samples, CLASS_NAMES,
                                   ds["name"], "Gabungan"))
        except Exception as e:
            print(f"  [ERROR gabungan] {e}")

        try:
            print("  -> evaluasi Model SATUAN")
            rows.append(eval_model(ds["satuan_path"], samples, dataset_classes,
                                   ds["name"], "Satuan"))
        except Exception as e:
            print(f"  [ERROR satuan] {e}")

        # Uji lintas-dataset: model satuan dari dataset lain diuji di test ini
        for em in ds.get("extra_models", []):
            try:
                print(f"  -> evaluasi {em['tipe']}")
                rows.append(eval_model(em["path"], samples, CLASS_NAMES,
                                       ds["name"], em["tipe"]))
            except Exception as e:
                print(f"  [ERROR {em['tipe']}] {e}")

    if rows:
        save_results(rows)
        print(f"\nSelesai. Hasil: {OUT_XLSX} & {OUT_CSV}")
    else:
        print("\n[ERROR] Tidak ada hasil. Cek path test_dir & model di KONFIGURASI.")


if __name__ == "__main__":
    main()
