"""
annotate_faithfulness.py
========================
Isi skor Faithfulness dari hasil_evaluasi_rag.json yang sudah ada.
TIDAK memanggil API lagi — hemat quota Gemini & resource Ollama.

3 Tier LLM yang dievaluasi:
  LOW    : Qwen2.5-3B     (Ollama lokal)
  MEDIUM : Gemini 2.5 Flash (Google API)
  HIGH   : Llama3.3-70B   (Ollama lokal)

Cara pakai:
  python annotate_faithfulness.py
  python annotate_faithfulness.py --input hasil_evaluasi_rag.json
"""

import json, argparse, numpy as np
from pathlib import Path

PANDUAN = """
Panduan skor Faithfulness:
  1.0 = Sangat akurat — semua fakta dan langkah sesuai ground truth
  0.8 = Akurat — satu-dua detail kurang spesifik, tidak menyesatkan
  0.5 = Sebagian benar — ada info penting terlewat atau berbeda
  0.2 = Sebagian besar salah atau tidak relevan
  0.0 = Tidak akurat / jawaban tidak sesuai penyakit yang ditanya
"""


def extract_generators(d: dict) -> list:
    """
    Handle dua format JSON yang mungkin:
      Format baru: {"generators": [ {llm: ..., per_query: ...}, ... ]}
      Format lama: {"generator": { "Ollama — xxx": {llm: ..., per_query: ...}, ... }}
    """
    if "generators" in d and isinstance(d["generators"], list):
        return d["generators"]

    if "generator" in d and isinstance(d["generator"], dict):
        result = []
        for llm_name, gen_data in d["generator"].items():
            if isinstance(gen_data, dict):
                result.append(gen_data)
        return result

    return []


def annotate(input_path: str):
    raw = Path(input_path).read_text(encoding="utf-8")
    d   = json.loads(raw)

    generators = extract_generators(d)

    if not generators:
        print("❌ Tidak ada data generator di file JSON.")
        print("   Pastikan kamu sudah menjalankan:")
        print("   python evaluate_rag.py --k 3 --llm all --output hasil_evaluasi_rag.json")
        return

    print(PANDUAN)
    print(f"Ditemukan {len(generators)} LLM untuk dianotasi.\n")

    for gen in generators:
        if "error" in gen:
            print(f"⚠️  Lewati {gen.get('llm','?')} karena ada error.")
            continue

        llm_name  = gen.get("llm", "Unknown LLM")
        per_query = gen.get("per_query", [])
        scores    = []

        already_done = [pq for pq in per_query if pq.get("faithfulness") is not None]
        if already_done:
            print(f"\n⚠️  {llm_name}: sudah ada {len(already_done)} skor faithfulness.")
            redo = input("   Isi ulang dari awal? (y/n): ").strip().lower()
            if redo != "y":
                scores = [pq["faithfulness"] for pq in per_query
                          if pq.get("faithfulness") is not None]
                gen["avg_faithfulness"] = round(float(np.mean(scores)), 4)
                print(f"   → Avg Faithfulness lama dipertahankan: {gen['avg_faithfulness']}")
                continue

        print(f"\n{'='*65}")
        print(f"  LLM: {llm_name}")
        print(f"  {len(per_query)} query perlu dinilai")
        print(f"{'='*65}")

        for i, pq in enumerate(per_query):
            generated = pq.get("generated", "")
            if not generated:
                print(f"\n[{i+1}/{len(per_query)}] {pq['id']} — {pq['disease']} → SKIP (jawaban kosong)")
                pq["faithfulness"] = 0.0
                scores.append(0.0)
                continue

            print(f"\n[{i+1}/{len(per_query)}] {pq['id']} — {pq['disease']}")
            print(f"\n  GROUND TRUTH:")
            print(f"  {pq['ground_truth']}")
            print(f"\n  JAWABAN LLM ({llm_name}):")
            print(f"  {generated[:600]}{'...' if len(generated) > 600 else ''}")
            print(f"\n  Cosine Similarity (otomatis): {pq.get('cosine_sim', 'N/A')}")

            while True:
                try:
                    val = input("\n  Skor Faithfulness (0.0 / 0.2 / 0.5 / 0.8 / 1.0): ").strip()
                    s   = float(val)
                    if 0.0 <= s <= 1.0:
                        break
                    print("  ⚠️  Harus antara 0.0 dan 1.0")
                except ValueError:
                    print("  ⚠️  Input tidak valid, ketik angka seperti 0.8")

            pq["faithfulness"] = round(s, 1)
            scores.append(s)
            print(f"  ✓ Tersimpan: {s}")

        if scores:
            avg = round(float(np.mean(scores)), 4)
            gen["avg_faithfulness"] = avg
            print(f"\n  ✅ Avg Faithfulness {llm_name}: {avg}")

    # Simpan kembali — pertahankan format asli
    if "generator" in d and isinstance(d["generator"], dict):
        for gen in generators:
            llm_name = gen.get("llm")
            if llm_name and llm_name in d["generator"]:
                d["generator"][llm_name] = gen
    else:
        d["generators"] = generators

    Path(input_path).write_text(
        json.dumps(d, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n{'='*65}")
    print(f"✅ Faithfulness disimpan ke: {input_path}")
    print_summary(d, generators)


def print_summary(d: dict, generators: list):
    ret = d.get("retriever", {})
    k   = ret.get("k", "?")

    print(f"\n{'='*65}")
    print("  RINGKASAN HASIL — salin ke laporan skripsi")
    print(f"{'='*65}")

    print(f"\n  [RETRIEVER] all-MiniLM-L6-v2 + FAISS | k={k} | n=18 query")
    print(f"  MRR            : {ret.get('MRR', 'N/A')}")
    print(f"  Recall@{k}       : {ret.get(f'Recall@{k}', 'N/A')}")
    print(f"  Precision@{k}    : {ret.get(f'Precision@{k}', 'N/A')}")

    print(f"\n  [GENERATOR] max_tokens=1024 | 3 Tier LLM")
    for gen in generators:
        if "error" in gen:
            continue
        print(f"\n  {gen['llm']}")
        print(f"    Cosine Similarity : {gen.get('avg_cosine_similarity', 'N/A')}")
        print(f"    ROUGE-1 F1        : {gen.get('avg_rouge1_f', 'N/A')}")
        print(f"    ROUGE-2 F1        : {gen.get('avg_rouge2_f', 'N/A')}")
        print(f"    ROUGE-L F1        : {gen.get('avg_rougeL_f', 'N/A')}")
        print(f"    Avg Time          : {gen.get('avg_generation_time_s', 'N/A')} detik")
        print(f"    Faithfulness      : {gen.get('avg_faithfulness', 'belum diisi')}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hasil_evaluasi_rag.json",
                        help="Path ke file JSON hasil evaluasi (default: hasil_evaluasi_rag.json)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ File tidak ditemukan: {args.input}")
        print(f"   Jalankan dulu: python evaluate_rag.py --k 3 --llm all")
        return

    annotate(args.input)


if __name__ == "__main__":
    main()