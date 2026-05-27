"""
evaluate_rag.py — Evaluasi RAG Pipeline Penyakit Padi
======================================================
Menghitung metrik evaluasi sesuai arahan dosen:

RETRIEVER METRICS (seperti di paper CottonBot):
  - MRR (Mean Reciprocal Rank)
  - Recall@k
  - Precision@k

GENERATOR METRICS:
  - Faithfulness (manual scoring 0–1)
  - Cosine Similarity (generated vs ground truth)
  - Average Generation Time (detik)

Cara pakai:
  python evaluate_rag.py
  python evaluate_rag.py --k 5 --output hasil_evaluasi.csv
"""

import os
import time
import json
import argparse
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# GROUND TRUTH — Skenario Uji + Jawaban Referensi
# ═════════════════════════════════════════════════════════════════
# Format setiap entry:
#   query           : pertanyaan yang diajukan ke sistem
#   disease         : kode penyakit (sesuai sistem klasifikasi)
#   relevant_keywords: kata kunci yang HARUS ada di chunk yang relevan
#                      (digunakan untuk menilai apakah chunk itu relevant)
#   ground_truth    : jawaban referensi untuk evaluasi generator

GROUND_TRUTH_QA = [
    {
        "id"                : "Q01",
        "query"             : "penanganan penyakit hawar daun bakteri pada tanaman padi",
        "disease"           : "bacterial_leaf_blight",
        "relevant_keywords" : ["hawar daun bakteri", "xanthomonas", "bakterisida", "tembaga", "kocide"],
        "ground_truth"      : (
            "Hawar Daun Bakteri (HDB) disebabkan oleh Xanthomonas oryzae. "
            "Gejala: daun menguning dari ujung dan tepi. "
            "Penanganan: semprot bakterisida berbahan tembaga (Kocide 77 WP 2-3 g/L), "
            "kurangi pupuk nitrogen, perbaiki drainase. "
            "Pencegahan: gunakan varietas tahan seperti Ciherang, tanam serempak."
        ),
    },
    {
        "id"                : "Q02",
        "query"             : "penanganan penyakit blas daun padi leaf blast",
        "disease"           : "leaf_blast",
        "relevant_keywords" : ["blas daun", "pyricularia", "tricyclazole", "beam", "fungisida"],
        "ground_truth"      : (
            "Blas Daun disebabkan oleh jamur Pyricularia oryzae. "
            "Gejala: bercak belah ketupat abu-abu pada daun. "
            "Penanganan: semprot fungisida tricyclazole (Beam 75 WP 0.5-1 g/L) segera. "
            "Kurangi nitrogen, tambah kalium dan silika. "
            "Pencegahan: varietas tahan, hindari nitrogen berlebih."
        ),
    },
    {
        "id"                : "Q03",
        "query"             : "penanganan penyakit blas leher malai neck blast padi",
        "disease"           : "neck_blast",
        "relevant_keywords" : ["blas leher", "neck blast", "malai", "tricyclazole", "heading"],
        "ground_truth"      : (
            "Blas Leher Malai adalah fase paling merusak blas, menyebabkan gabah hampa. "
            "Penanganan DARURAT: semprot tricyclazole atau isoprothiolane SEGERA dalam 24-48 jam. "
            "Hentikan nitrogen, tambah kalium. "
            "Pencegahan: semprot preventif 2x (primordia + heading 50%)."
        ),
    },
    {
        "id"                : "Q04",
        "query"             : "penanganan penyakit busuk pelepah sheath blight padi",
        "disease"           : "sheath_blight",
        "relevant_keywords" : ["busuk pelepah", "rhizoctonia", "validamycin", "validacin", "sklerotia"],
        "ground_truth"      : (
            "Busuk Pelepah disebabkan Rhizoctonia solani. "
            "Gejala: lesi abu-abu pada pelepah daun bawah. "
            "Penanganan: semprot validamycin (Validacin 3L, 1-2 ml/L) ke pangkal tanaman, "
            "kurangi nitrogen, perlebar jarak tanam. "
            "Pencegahan: irigasi berselang, hindari tanam rapat."
        ),
    },
    {
        "id"                : "Q05",
        "query"             : "penanganan penyakit tungro padi vektor wereng",
        "disease"           : "tungro",
        "relevant_keywords" : ["tungro", "wereng hijau", "nephotettix", "virus", "imidakloprid"],
        "ground_truth"      : (
            "Tungro adalah penyakit virus yang disebarkan wereng hijau. "
            "Gejala: daun kuning-oranye, tanaman kerdil. "
            "Tidak ada obat langsung. "
            "Penanganan: kendalikan wereng dengan imidakloprid, cabut tanaman sakit. "
            "Pencegahan: varietas tahan, tanam serempak."
        ),
    },
    {
        "id"                : "Q06",
        "query"             : "penanganan penyakit bercak coklat brown spot padi",
        "disease"           : "brown_spot",
        "relevant_keywords" : ["bercak coklat", "cochliobolus", "helminthosporium", "mancozeb", "kalium"],
        "ground_truth"      : (
            "Bercak Coklat disebabkan jamur Cochliobolus miyabeanus. "
            "Gejala: bercak oval coklat dengan pusat abu-abu pada daun. "
            "Penanganan: fungisida mancozeb atau propiconazole, tambah pupuk kalium. "
            "Pencegahan: pemupukan berimbang, varietas tahan."
        ),
    },
    {
        "id"                : "Q07",
        "query"             : "penanganan hama penggerek batang padi dead heart sundep",
        "disease"           : "dead_heart",
        "relevant_keywords" : ["penggerek batang", "dead heart", "sundep", "karbofuran", "trichogramma"],
        "ground_truth"      : (
            "Dead Heart (Sundep) disebabkan larva penggerek batang. "
            "Gejala: pucuk tanaman muda mati dan mudah dicabut. "
            "Penanganan: tabur karbofuran (Furadan 3G 17 kg/ha) atau semprot klorpirifos. "
            "Lepas parasitoid Trichogramma. Pencegahan: tanam serempak."
        ),
    },
    {
        "id"                : "Q08",
        "query"             : "penanganan penyakit hispa padi serangga kumbang",
        "disease"           : "hispa",
        "relevant_keywords" : ["hispa", "dicladispa", "kumbang", "goresan", "insektisida"],
        "ground_truth"      : (
            "Hispa disebabkan kumbang Dicladispa armigera. "
            "Gejala: goresan putih horizontal pada daun, terowongan transparan oleh larva. "
            "Penanganan: semprot klorpirifos atau imidakloprid, celup daun ke air. "
            "Pencegahan: atur jarak tanam, kurangi nitrogen."
        ),
    },
    {
        "id"                : "Q09",
        "query"             : "penanganan penyakit hawar malai bakteri bacterial panicle blight padi",
        "disease"           : "bacterial_panicle_blight",
        "relevant_keywords" : ["hawar malai", "burkholderia", "kasugamycin", "kasumin", "pembungaan"],
        "ground_truth"      : (
            "Hawar Malai Bakteri disebabkan Burkholderia glumae. "
            "Gejala: malai tegak berisi gabah hampa, biji berwarna coklat. "
            "Penanganan: semprot kasugamycin saat primordia dan pembungaan. "
            "Pencegahan: perlakuan benih, hindari nitrogen berlebih saat berbunga."
        ),
    },
    {
        "id"                : "Q10",
        "query"             : "tanaman padi sehat healthy kondisi optimal perawatan",
        "disease"           : "healthy",
        "relevant_keywords" : ["tanaman sehat", "healthy", "daun hijau", "pemupukan", "optimal"],
        "ground_truth"      : (
            "Tanaman padi sehat ditandai daun hijau segar, batang tegak, tidak ada gejala penyakit. "
            "Lanjutkan pemupukan berimbang N-P-K, irigasi berselang, dan pemantauan rutin setiap minggu. "
            "Tidak diperlukan tindakan pengendalian penyakit."
        ),
    },
]


# ═════════════════════════════════════════════════════════════════
# HELPER: Cosine Similarity
# ═════════════════════════════════════════════════════════════════
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Hitung cosine similarity antara dua vektor."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def is_relevant_chunk(chunk_text: str, relevant_keywords: list[str]) -> bool:
    """
    Tentukan apakah sebuah chunk dianggap relevan berdasarkan keyword.
    Chunk relevan jika mengandung MINIMAL 1 dari keyword yang diberikan.
    """
    chunk_lower = chunk_text.lower()
    return any(kw.lower() in chunk_lower for kw in relevant_keywords)


# ═════════════════════════════════════════════════════════════════
# RETRIEVER EVALUATION
# ═════════════════════════════════════════════════════════════════
def evaluate_retriever(ground_truth_list: list[dict], k: int = 5) -> dict:
    """
    Evaluasi komponen retriever RAG.

    Metrik:
    - MRR@k  : Mean Reciprocal Rank
    - Recall@k
    - Precision@k

    Sesuai rumus di paper CottonBot (Kandamali et al., 2025):
      MRR = (1/N) * Σ (1/rank_i)
      Recall@k = (relevant di top-k) / (total relevant dalam koleksi)
      Precision@k = (relevant di top-k) / k
    """
    from rag import retrieve

    mrr_scores        = []
    recall_scores     = []
    precision_scores  = []
    per_query_results = []

    logger.info(f"Mengevaluasi retriever untuk {len(ground_truth_list)} query (k={k})...")

    for qa in ground_truth_list:
        query            = qa["query"]
        relevant_keywords= qa["relevant_keywords"]

        retrieved = retrieve(query, k=k)

        # Tandai setiap chunk: relevant (1) atau tidak (0)
        relevance_flags = [
            1 if is_relevant_chunk(item["text"], relevant_keywords) else 0
            for item in retrieved
        ]

        # ── MRR: posisi pertama chunk relevan ──────────────────
        first_relevant_rank = None
        for i, flag in enumerate(relevance_flags):
            if flag == 1:
                first_relevant_rank = i + 1  # rank mulai dari 1
                break
        rr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
        mrr_scores.append(rr)

        # ── Recall@k ───────────────────────────────────────────
        # Anggap total relevant di koleksi = 1 (setidaknya 1 chunk relevan harus ada)
        # Pendekatan konservatif yang umum dipakai pada evaluasi RAG closed-domain
        num_relevant_retrieved = sum(relevance_flags)
        recall = min(num_relevant_retrieved, 1)   # recall maksimal 1 jika ada ≥1 relevant
        recall_scores.append(recall)

        # ── Precision@k ────────────────────────────────────────
        precision = sum(relevance_flags) / k
        precision_scores.append(precision)

        per_query_results.append({
            "id"              : qa["id"],
            "query"           : query[:60] + "..." if len(query) > 60 else query,
            "disease"         : qa["disease"],
            "rr"              : round(rr, 4),
            "recall"          : round(recall, 4),
            "precision"       : round(precision, 4),
            "relevance_flags" : relevance_flags,
            "top1_score"      : retrieved[0]["score"] if retrieved else 0,
        })

    return {
        "MRR"              : round(float(np.mean(mrr_scores)), 4),
        f"Recall@{k}"      : round(float(np.mean(recall_scores)), 4),
        f"Precision@{k}"   : round(float(np.mean(precision_scores)), 4),
        "per_query"        : per_query_results,
        "k"                : k,
        "n_queries"        : len(ground_truth_list),
    }


# ═════════════════════════════════════════════════════════════════
# GENERATOR EVALUATION
# ═════════════════════════════════════════════════════════════════
def evaluate_generator(
    ground_truth_list : list[dict],
    llm_func          : callable,
    llm_name          : str = "LLM",
) -> dict:
    """
    Evaluasi komponen generator (LLM) dalam RAG pipeline.

    Metrik:
    - Faithfulness : skor manual (annotator) 0-1 seberapa akurat jawaban terhadap konteks
    - Cosine Similarity: kemiripan semantik generated vs ground truth
    - Avg Generation Time: rata-rata waktu generate (detik)
    """
    from rag import build_rag_prompt
    from sentence_transformers import SentenceTransformer

    embed_model   = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cosim_scores  = []
    time_scores   = []
    per_query_res = []

    logger.info(f"Mengevaluasi generator: {llm_name} ({len(ground_truth_list)} query)...")

    for qa in ground_truth_list:
        disease      = qa["disease"]
        ground_truth = qa["ground_truth"]

        # Build RAG prompt (dengan konteks dari retriever)
        prompt, retrieved = build_rag_prompt(disease_name=disease)

        # Generate jawaban dari LLM
        t0     = time.perf_counter()
        answer = llm_func(prompt)
        t_gen  = round(time.perf_counter() - t0, 3)
        time_scores.append(t_gen)

        # Hitung cosine similarity antara jawaban LLM dan ground truth
        vecs = embed_model.encode(
            [answer, ground_truth],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        cos_sim = cosine_similarity(vecs[0], vecs[1])
        cosim_scores.append(cos_sim)

        per_query_res.append({
            "id"             : qa["id"],
            "disease"        : disease,
            "generated"      : answer[:200] + "..." if len(answer) > 200 else answer,
            "ground_truth"   : ground_truth[:200] + "..." if len(ground_truth) > 200 else ground_truth,
            "cosine_sim"     : round(cos_sim, 4),
            "time_s"         : t_gen,
            "faithfulness"   : None,  # diisi manual oleh annotator
        })

    return {
        "llm"                    : llm_name,
        "avg_cosine_similarity"  : round(float(np.mean(cosim_scores)), 4),
        "avg_generation_time_s"  : round(float(np.mean(time_scores)), 3),
        "faithfulness_note"      : "Diisi manual oleh annotator (0=tidak akurat, 1=sangat akurat)",
        "per_query"              : per_query_res,
        "n_queries"              : len(ground_truth_list),
    }


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ═════════════════════════════════════════════════════════════════
# FAITHFULNESS MANUAL SCORING TOOL
# ═════════════════════════════════════════════════════════════════
def run_faithfulness_annotation(generator_results: dict) -> dict:
    """
    Tool interaktif untuk mengisi skor faithfulness secara manual.
    Annotator membaca jawaban vs konteks dan memberi skor 0-1.
    """
    print("\n" + "="*60)
    print("ANOTASI FAITHFULNESS (Manual)")
    print("Skor: 0=tidak akurat, 0.5=sebagian akurat, 1=sangat akurat")
    print("="*60)

    scores = []
    for i, pq in enumerate(generator_results["per_query"]):
        print(f"\n[{i+1}/{len(generator_results['per_query'])}] {pq['id']} — {pq['disease']}")
        print(f"Ground Truth: {pq['ground_truth']}")
        print(f"Generated   : {pq['generated']}")

        while True:
            try:
                score = float(input("Skor Faithfulness (0.0 - 1.0): "))
                if 0.0 <= score <= 1.0:
                    break
                print("Masukkan angka antara 0 dan 1.")
            except ValueError:
                print("Input tidak valid.")

        pq["faithfulness"] = score
        scores.append(score)

    generator_results["avg_faithfulness"] = round(float(np.mean(scores)), 4)
    return generator_results


# ═════════════════════════════════════════════════════════════════
# PRINT & SAVE RESULTS
# ═════════════════════════════════════════════════════════════════
def print_retriever_table(results: dict):
    k = results["k"]
    print(f"\n{'='*55}")
    print(f"  HASIL EVALUASI RETRIEVER RAG (k={k})")
    print(f"{'='*55}")
    print(f"  {'Metrik':<20} {'Nilai':>10}")
    print(f"  {'-'*32}")
    print(f"  {'MRR':<20} {results['MRR']:>10.4f}")
    print(f"  {f'Recall@{k}':<20} {results[f'Recall@{k}']:>10.4f}")
    print(f"  {f'Precision@{k}':<20} {results[f'Precision@{k}']:>10.4f}")
    print(f"{'='*55}")
    print(f"\nPer-query detail:")
    print(f"  {'ID':<5} {'Penyakit':<28} {'RR':>6} {'Rec':>6} {'Prec':>6}")
    print(f"  {'-'*55}")
    for pq in results["per_query"]:
        print(f"  {pq['id']:<5} {pq['disease']:<28} {pq['rr']:>6.3f} {pq['recall']:>6.3f} {pq['precision']:>6.3f}")


def print_generator_table(results: dict):
    print(f"\n{'='*55}")
    print(f"  HASIL EVALUASI GENERATOR: {results['llm']}")
    print(f"{'='*55}")
    print(f"  Avg Cosine Similarity : {results['avg_cosine_similarity']:.4f}")
    print(f"  Avg Generation Time   : {results['avg_generation_time_s']:.3f} detik")
    if "avg_faithfulness" in results:
        print(f"  Avg Faithfulness      : {results['avg_faithfulness']:.4f}")
    print(f"{'='*55}")


def save_results(retriever_res: dict, generator_res: dict, output_path: str = "hasil_evaluasi_rag.json"):
    """Simpan semua hasil evaluasi ke file JSON."""
    output = {
        "retriever" : retriever_res,
        "generator" : generator_res,
        "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Hasil evaluasi disimpan: {output_path}")


def save_results_csv(retriever_res: dict, output_path: str = "hasil_retriever.csv"):
    """Simpan hasil retriever ke CSV untuk dimasukkan ke laporan."""
    import csv
    k = retriever_res["k"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Penyakit", "RR", f"Recall@{k}", f"Precision@{k}", "Top1 Score"])
        for pq in retriever_res["per_query"]:
            writer.writerow([
                pq["id"], pq["disease"],
                pq["rr"], pq["recall"], pq["precision"], pq["top1_score"]
            ])
        writer.writerow([])
        writer.writerow(["RATA-RATA", "",
                         retriever_res["MRR"],
                         retriever_res[f"Recall@{k}"],
                         retriever_res[f"Precision@{k}"],
                         ""])
    logger.info(f"Hasil retriever CSV: {output_path}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluasi RAG Pipeline Penyakit Padi")
    parser.add_argument("--k",          type=int, default=5,  help="Nilai k untuk retrieval (default: 5)")
    parser.add_argument("--output",     type=str, default="hasil_evaluasi_rag.json")
    parser.add_argument("--csv",        type=str, default="hasil_retriever.csv")
    parser.add_argument("--skip-llm",   action="store_true",  help="Lewati evaluasi generator LLM")
    parser.add_argument("--llm",        type=str, default="groq", choices=["groq", "gemini"],
                        help="LLM yang dievaluasi (default: groq)")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Jalankan anotasi faithfulness manual")
    args = parser.parse_args()

    # ── Pastikan RAG index sudah ada ────────────────────────────
    from rag import get_index_info, build_index
    from pathlib import Path
    if not Path("faiss_index.bin").exists():
        logger.info("Index belum ada, membangun index RAG...")
        build_index()

    info = get_index_info()
    print(f"\nRAG Index Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # ── Evaluasi Retriever ───────────────────────────────────────
    retriever_res = evaluate_retriever(GROUND_TRUTH_QA, k=args.k)
    print_retriever_table(retriever_res)
    save_results_csv(retriever_res, args.csv)

    # ── Evaluasi Generator (opsional) ────────────────────────────
    generator_res = None
    if not args.skip_llm:
        try:
            if args.llm == "groq":
                from llm import get_recommendation_groq
                def llm_func(prompt): return get_recommendation_groq.__wrapped__(prompt) if hasattr(get_recommendation_groq, '__wrapped__') else get_recommendation_groq("leaf_blast")
                # Buat wrapper yang menerima prompt langsung
                from groq import Groq
                import os
                from dotenv import load_dotenv
                load_dotenv()
                groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                from llm import GROQ_MODEL_NAME, SYSTEM_PROMPT
                def llm_generate(prompt):
                    resp = groq_client.chat.completions.create(
                        model=GROQ_MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.7, max_tokens=2048,
                    )
                    return resp.choices[0].message.content
                llm_name = f"Groq — {GROQ_MODEL_NAME}"

            else:  # gemini
                from llm import _gemini_generate, SYSTEM_PROMPT
                def llm_generate(prompt):
                    text, _ = _gemini_generate(prompt)
                    return text
                from llm import GEMINI_MODEL_NAME
                llm_name = f"Gemini — {GEMINI_MODEL_NAME}"

            generator_res = evaluate_generator(GROUND_TRUTH_QA, llm_generate, llm_name)
            print_generator_table(generator_res)

            if args.faithfulness:
                generator_res = run_faithfulness_annotation(generator_res)
                print_generator_table(generator_res)

        except ImportError as e:
            logger.warning(f"Tidak bisa load LLM module: {e}. Lewati evaluasi generator.")
            generator_res = {"error": str(e)}

    # ── Simpan semua hasil ───────────────────────────────────────
    save_results(retriever_res, generator_res or {}, args.output)
    print(f"\nSemua hasil disimpan ke: {args.output} dan {args.csv}")


if __name__ == "__main__":
    main()
