#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_realcase.py — Uji REAL-CASE: GABUNGAN vs SEMUA SATUAN (swin_base)
================================================================================
Untuk data KECIL (1-3 gambar per kelas). Ini STUDI KASUS / DEMONSTRASI lapangan,
BUKAN akurasi statistik test-set penuh.

Model yang diuji (semua swin_base, sama seperti benchmark_model.py):
  - Gabungan            (14 kelas)
  - Satuan paddy_v3     (9 kelas)
  - Satuan Citra_Daun_Padi        (3 kelas)   ← lintas
  - Satuan JENIS_PENYAKIT_PADI    (3 kelas)   ← lintas
  - Satuan Paddy_disease          (11 kelas)  ← lintas

Output (sesuai data kecil):
  - Tabel prediksi PER-GAMBAR per model (aktual vs prediksi + confidence)
  - Ringkasan "benar X dari Y" per model + PLAFON (kelas yang dikenali model)
  - Tabel perbandingan antar-model
  (F1/precision/recall macro sengaja tidak ditonjolkan: tak bermakna pd 1-3 sampel.)

Struktur folder (per kelas):
  train model/
    ├── bacterial_leaf_blight/ *.jpg
    ├── leaf_blast/ *.jpg
    └── ... (14 kelas)

Jalankan (dari folder API, sejajar benchmark_model.py):
  python benchmark_realcase.py
  python benchmark_realcase.py --dir "PATH/ke/folder"
"""

import os
import argparse

import torch
from PIL import Image

# Pakai ulang logika teruji dari benchmark_model.py
# (impor aman: main() di sana hanya jalan lewat __main__).
from benchmark_model import (
    CLASS_NAMES, to_canonical, load_model, get_transform, DEVICE, IMG_EXTS,
)

# ===== KONFIGURASI DEFAULT =====
REALCASE_DIR = r"D:\coolyeah\coolyeah-kok-gini-amat\apa_ini_tetiba_sem_8_aja\skripsi_ketar_ketir\skripsi-fix\train model"
OUT_XLSX = "hasil_realcase.xlsx"
OUT_CSV = "hasil_realcase.csv"

# Semua model swin_base: gabungan + 4 satuan per-dataset.
MODELS = [
    {"name": "Gabungan",              "path": "model_gabungan/swin_base_best.h5"},
    {"name": "Satuan-paddy_v3",       "path": "models_perdataset/swin_base_paddy-dataset-v3-augmentasi_best.h5"},
    {"name": "Satuan-Citra_Daun_Padi",     "path": "models_perdataset/swin_base_Citra_Daun_Padi_best.h5"},
    {"name": "Satuan-JENIS_PENYAKIT_PADI", "path": "models_perdataset/swin_base_JENIS_PENYAKIT_PADI_best.h5"},
    {"name": "Satuan-Paddy_disease",       "path": "models_perdataset/swin_base_Paddy-disease-classification_best.h5"},
]


@torch.no_grad()
def predict_one(model, img_tensor, class_list):
    x = img_tensor.unsqueeze(0).to(DEVICE)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    prob = torch.softmax(out, dim=1)[0]
    conf, idx = torch.max(prob, dim=0)
    j = int(idx)
    raw = class_list[j] if j < len(class_list) else str(j)
    return to_canonical(raw), float(conf) * 100.0


def gather_samples(root, transform):
    """Baca gambar sekali, transform sekali (dipakai ulang utk semua model)."""
    samples = []
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        true_canon = to_canonical(cls)
        for fn in sorted(os.listdir(cdir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            path = os.path.join(cdir, fn)
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"  [skip] {fn}: {e}")
                continue
            samples.append({
                "true": true_canon, "folder": cls, "file": fn,
                "tensor": transform(img),
            })
    return samples


def eval_model(m, samples):
    """Return dict hasil evaluasi 1 model atas semua sample."""
    if not os.path.exists(m["path"]):
        print(f"[skip] model tidak ditemukan: {m['path']}")
        return None
    model, ncls, ckpt_classes, _ = load_model(m["path"])
    if ckpt_classes and len(ckpt_classes) == ncls:
        class_list = [to_canonical(c) for c in ckpt_classes]
    else:
        class_list = CLASS_NAMES
    known = set(class_list)

    rows = []
    benar = 0
    plafon = 0   # jml gambar yg kelas-nya memang dikenali model (ceiling)
    for s in samples:
        pred, conf = predict_one(model, s["tensor"], class_list)
        ok = (pred == s["true"])
        if s["true"] in known:
            plafon += 1
        if ok:
            benar += 1
        rows.append({
            "model": m["name"], "aktual": s["true"], "prediksi": pred,
            "confidence_%": round(conf, 1), "benar": "YA" if ok else "TIDAK",
            "file": s["file"],
        })
    return {"name": m["name"], "ncls": ncls, "rows": rows,
            "benar": benar, "total": len(samples), "plafon": plafon}


def main():
    ap = argparse.ArgumentParser(description="Uji real-case: gabungan vs semua satuan swin_base")
    ap.add_argument("--dir", default=REALCASE_DIR, help="folder real-case (subfolder per kelas)")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[ERROR] Folder tidak ditemukan: {args.dir}")
        return

    print(f"Device: {DEVICE}")
    print(f"Folder: {args.dir}\n")

    transform = get_transform()
    samples = gather_samples(args.dir, transform)
    if not samples:
        print("[ERROR] Tidak ada gambar. Pastikan ada subfolder per kelas berisi gambar.")
        return
    print(f"Total gambar real-case: {len(samples)}\n")

    results = []
    for m in MODELS:
        print(f"=> evaluasi {m['name']} ...")
        r = eval_model(m, samples)
        if r is not None:
            results.append(r)

    if not results:
        print("[ERROR] Tidak ada model yang berhasil dievaluasi.")
        return

    # ===== Tabel per-gambar per model =====
    for r in results:
        print(f"\n================ PREDIKSI PER-GAMBAR: {r['name']} "
              f"({r['ncls']} kelas) ================")
        print(f"{'aktual':<26}{'prediksi':<26}{'conf%':>7}  {'benar':<6} file")
        print("-" * 92)
        for row in r["rows"]:
            print(f"{row['aktual']:<26}{row['prediksi']:<26}"
                  f"{row['confidence_%']:>7}  {row['benar']:<6} {row['file']}")
        print(f"-> benar {r['benar']}/{r['total']} "
              f"({r['benar']/r['total']*100:.1f}%) | plafon (kelas dikenali): "
              f"{r['plafon']}/{r['total']}")

    # ===== Tabel perbandingan antar-model =====
    print("\n================ PERBANDINGAN ANTAR-MODEL (REAL-CASE) ================")
    print(f"{'model':<28}{'n_kelas':>8}{'benar':>8}{'total':>7}{'akurasi':>10}{'plafon':>9}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<28}{r['ncls']:>8}{r['benar']:>8}{r['total']:>7}"
              f"{r['benar']/r['total']*100:>9.1f}%{r['plafon']:>8}")
    print("\nCatatan: model satuan hanya bisa benar utk gambar yg kelasnya ada di\n"
          "dataset latihnya (lihat kolom 'plafon'). Ini normal, bukan model rusak.")

    # ===== Simpan =====
    try:
        import pandas as pd
        all_rows = [row for r in results for row in r["rows"]]
        df = pd.DataFrame(all_rows, columns=["model", "aktual", "prediksi",
                                             "confidence_%", "benar", "file"])
        df.to_csv(OUT_CSV, index=False)
        try:
            with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
                ringkas = pd.DataFrame([{
                    "model": r["name"], "n_kelas": r["ncls"], "benar": r["benar"],
                    "total": r["total"],
                    "akurasi_%": round(r["benar"] / r["total"] * 100, 1),
                    "plafon": r["plafon"],
                } for r in results])
                ringkas.to_excel(w, sheet_name="ringkasan", index=False)
                for r in results:
                    sheet = r["name"][:31]
                    pd.DataFrame(r["rows"]).to_excel(w, sheet_name=sheet, index=False)
        except Exception as e:
            print(f"[WARN] gagal tulis xlsx: {e} (CSV tetap tersimpan)")
        print(f"\nSelesai. Hasil: {OUT_XLSX} & {OUT_CSV}")
    except Exception as e:
        print(f"[WARN] gagal simpan file: {e}")


if __name__ == "__main__":
    main()
