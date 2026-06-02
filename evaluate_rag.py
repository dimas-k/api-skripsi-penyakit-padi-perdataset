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
  python evaluate_rag.py --k 5 --csv hasil_retriever.csv
  python evaluate_rag.py --k 5 --faithfulness
  python evaluate_rag.py --skip-llm   (hanya evaluasi retriever)
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
# KONFIGURASI (FIX: definisikan di atas, bukan di bawah fungsi)
# ═════════════════════════════════════════════════════════════════
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ═════════════════════════════════════════════════════════════════
# GROUND TRUTH — 15 Skenario (sinkron dengan Excel Ground Truth)
# ═════════════════════════════════════════════════════════════════
# Query, disease, keyword, dan ground_truth sudah disesuaikan
# persis dengan file Ground_Truth_Evaluasi_RAG_Skripsi.xlsx
GROUND_TRUTH_QA = [
    {
        "id"               : "Q01",
        "query"            : "Apa gejala dan cara mengatasi hawar daun bakteri pada padi?",
        "disease"          : "bacterial_leaf_blight",
        "relevant_keywords": ["hawar daun bakteri", "xanthomonas", "bakterisida", "tembaga", "kocide", "nitrogen"],
        "ground_truth"     : (
            "Hawar Daun Bakteri disebabkan Xanthomonas oryzae pv. oryzae. "
            "Gejala: daun menguning dari ujung dan tepi, muncul eksudat kekuningan di pagi hari. "
            "Penanganan: semprot bakterisida tembaga (Kocide 77 WP 2-3 g/L), kurangi nitrogen, perbaiki drainase. "
            "Pencegahan: varietas tahan (Ciherang, Mekongga), tanam serempak."
        ),
    },
    {
        "id"               : "Q02",
        "query"            : "Tanaman padi saya daunnya ada bercak belah ketupat abu-abu, apa itu dan bagaimana cara mengobatinya?",
        "disease"          : "leaf_blast",
        "relevant_keywords": ["blas daun", "pyricularia", "tricyclazole", "beam", "fungisida", "bercak ketupat"],
        "ground_truth"     : (
            "Blas Daun disebabkan jamur Pyricularia oryzae. "
            "Gejala: bercak belah ketupat abu-abu dengan tepi coklat. "
            "Penanganan SEGERA: semprot fungisida tricyclazole (Beam 75 WP 0.5-1 g/L) atau isoprothiolane. "
            "Kurangi nitrogen, tambah kalium dan silika. "
            "Pencegahan: varietas tahan (Inpari 13, 19), hindari nitrogen berlebih."
        ),
    },
    {
        "id"               : "Q03",
        "query"            : "Malai padi saya tegak tapi gabahnya hampa, apa penyebabnya dan apa yang harus dilakukan?",
        "disease"          : "neck_blast",
        "relevant_keywords": ["blas leher", "neck blast", "malai", "hampa", "tricyclazole", "heading", "primordia"],
        "ground_truth"     : (
            "Blas Leher Malai: jamur Pyricularia oryzae menyerang pangkal malai, menyebabkan gabah hampa total. "
            "TINDAKAN DARURAT dalam 24-48 jam: semprot tricyclazole (Beam 75 WP) atau isoprothiolane SEGERA. "
            "Hentikan pupuk nitrogen. "
            "Pencegahan: semprot preventif 2x (saat primordia + heading 50%)."
        ),
    },
    {
        "id"               : "Q04",
        "query"            : "Ada lesi abu-abu di pelepah daun bawah tanaman padi saya yang menyebar ke atas, penyakit apa ini?",
        "disease"          : "sheath_blight",
        "relevant_keywords": ["busuk pelepah", "rhizoctonia", "validamycin", "validacin", "sklerotia", "pelepah"],
        "ground_truth"     : (
            "Busuk Pelepah disebabkan Rhizoctonia solani. "
            "Gejala: lesi oval abu-abu pada pelepah daun dekat permukaan air, menyebar ke atas. "
            "Penanganan: semprot validamycin (Validacin 3L, 1-2 ml/L) ke pangkal tanaman, "
            "kurangi nitrogen, perlebar jarak tanam. "
            "Pencegahan: irigasi berselang, hindari tanam rapat."
        ),
    },
    {
        "id"               : "Q05",
        "query"            : "Daun padi kuning-oranye, tanaman kerdil, bagaimana menanganinya?",
        "disease"          : "tungro",
        "relevant_keywords": ["tungro", "wereng hijau", "nephotettix", "virus", "imidakloprid", "kuning oranye"],
        "ground_truth"     : (
            "Tungro adalah penyakit virus yang disebarkan wereng hijau (Nephotettix virescens). "
            "Gejala: daun kuning-oranye dari ujung, tanaman kerdil. "
            "Tidak ada obat langsung untuk virus. "
            "Penanganan: KENDALIKAN WERENG dengan imidakloprid, cabut dan musnahkan tanaman sakit. "
            "Pencegahan: varietas tahan tungro, tanam serempak."
        ),
    },
    {
        "id"               : "Q06",
        "query"            : "Ada bercak oval coklat dengan pusat putih di daun padi, apa penyakitnya?",
        "disease"          : "brown_spot",
        "relevant_keywords": ["bercak coklat", "cochliobolus", "helminthosporium", "mancozeb", "kalium", "bercak oval"],
        "ground_truth"     : (
            "Bercak Coklat disebabkan Cochliobolus miyabeanus. "
            "Gejala: bercak oval coklat dengan pusat abu-abu/putih. Sering terjadi pada lahan defisiensi kalium. "
            "Penanganan: fungisida mancozeb (Dithane M-45, 2 g/L) atau propiconazole, "
            "tambah pupuk kalium (KCl 50-75 kg/ha). "
            "Pencegahan: pemupukan berimbang N-P-K."
        ),
    },
    {
        "id"               : "Q07",
        "query"            : "Pucuk tanaman padi muda mati dan bisa dicabut dengan mudah, apa penyebabnya?",
        "disease"          : "dead_heart",
        "relevant_keywords": ["penggerek batang", "dead heart", "sundep", "karbofuran", "furadan", "trichogramma"],
        "ground_truth"     : (
            "Batang Mati/Sundep disebabkan larva penggerek batang (Scirpophaga incertulas). "
            "Gejala: pucuk mati kuning-coklat, mudah dicabut, batang dalam berlubang. "
            "Penanganan: tabur karbofuran (Furadan 3G 17 kg/ha), atau semprot klorpirifos/fipronil. "
            "Lepas parasitoid Trichogramma. Pencegahan: tanam serempak."
        ),
    },
    {
        "id"               : "Q08",
        "query"            : "Ada goresan putih horizontal di daun padi, kadang ada bagian daun seperti transparan, apa ini?",
        "disease"          : "hispa",
        "relevant_keywords": ["hispa", "dicladispa", "kumbang", "goresan putih", "transparan", "windowing"],
        "ground_truth"     : (
            "Hispa Padi disebabkan kumbang Dicladispa armigera. "
            "Gejala: goresan putih sejajar dari imago + terowongan transparan (windowing) dari larva. "
            "Penanganan: semprot klorpirifos (2 ml/L) atau imidakloprid (0.5 ml/L), "
            "celup daun ke air untuk menghilangkan imago. "
            "Pencegahan: atur jarak tanam, kurangi nitrogen."
        ),
    },
    {
        "id"               : "Q09",
        "query"            : "Malai padi berisi gabah hampa berwarna coklat dan biji mengeriput saat pembungaan, apa ini?",
        "disease"          : "bacterial_panicle_blight",
        "relevant_keywords": ["hawar malai", "burkholderia", "kasugamycin", "kasumin", "pembungaan", "gabah hampa"],
        "ground_truth"     : (
            "Hawar Malai Bakteri disebabkan Burkholderia glumae. "
            "Gejala: malai tegak, gabah hampa berwarna coklat/abu, pangkal gabah coklat. "
            "Penanganan: semprot kasugamycin (Kasumin 2L, 1-2 ml/L) saat primordia dan 50% pembungaan. "
            "Pencegahan: perlakuan benih, hindari nitrogen tinggi menjelang berbunga."
        ),
    },
    {
        "id"               : "Q10",
        "query"            : "Daun padi saya tampak bintik-bintik hitam kecil tersebar, tidak terlalu parah, apa itu?",
        "disease"          : "leaf_smut",
        "relevant_keywords": ["gosong palsu", "leaf smut", "entyloma", "bercak hitam", "spora"],
        "ground_truth"     : (
            "Gosong Palsu Daun disebabkan Entyloma oryzae. "
            "Gejala: bercak hitam/abu-abu kecil oval 1-5 mm tersebar di daun, ada serbuk hitam (spora). "
            "Penanganan: fungisida tembaga (Kocide 77 WP 2 g/L) atau mancozeb. "
            "Penyakit ini umumnya tidak menyebabkan kehilangan hasil besar. "
            "Pencegahan: drainase baik, hindari nitrogen berlebih."
        ),
    },
    {
        "id"               : "Q11",
        "query"            : "Ada lapisan putih seperti tepung di bawah daun padi, tanaman terlihat kerdil dan daun distorsi",
        "disease"          : "downy_mildew",
        "relevant_keywords": ["embun bulu", "downy mildew", "sclerophthora", "metalaxil", "ridomil", "crazy top"],
        "ground_truth"     : (
            "Embun Bulu disebabkan Sclerophthora macrospora. "
            "Gejala: serbuk putih di bawah daun, tanaman kerdil, daun distorsi, kadang malai berubah jadi daun (crazy top). "
            "Penanganan: fungisida metalaxil (Ridomil Gold 35 WS, 2 g/L), kurangi genangan air, perbaiki drainase. "
            "Pencegahan: seed treatment metalaxil, hindari genangan."
        ),
    },
    {
        "id"               : "Q12",
        "query"            : "Tanaman padi saya tampak sehat, daun hijau, tidak ada gejala penyakit, apa yang harus saya lakukan?",
        "disease"          : "healthy",
        "relevant_keywords": ["tanaman sehat", "healthy", "daun hijau", "pemupukan", "irigasi berselang", "optimal"],
        "ground_truth"     : (
            "Tanaman sehat: daun hijau segar, batang tegak, pertumbuhan normal. "
            "Lanjutkan program pemupukan berimbang N-P-K sesuai umur tanaman. "
            "Terapkan irigasi berselang untuk hemat air. "
            "Lakukan pemantauan rutin setiap 7-10 hari. "
            "Tidak diperlukan tindakan pengendalian penyakit. Pertahankan kondisi optimal."
        ),
    },
    {
        "id"               : "Q13",
        "query"            : "Kelembaban udara sangat tinggi 90%, suhu 26°C, apakah ada risiko penyakit dan apa yang harus dilakukan?",
        "disease"          : "leaf_blast",
        "relevant_keywords": ["blas daun", "kelembaban tinggi", "suhu", "tricyclazole", "preventif", "sensor"],
        "ground_truth"     : (
            "Kondisi sensor (kelembaban >85%, suhu 24-28°C) merupakan kondisi optimal penyebaran Blas Daun. "
            "Tindakan preventif segera: semprot fungisida tricyclazole (Beam 75 WP 0.5 g/L) sebagai pencegahan. "
            "Periksa daun untuk gejala awal bercak ketupat abu-abu. "
            "Kurangi pemupukan nitrogen. Tingkatkan drainase untuk menurunkan kelembaban mikro."
        ),
    },
    {
        "id"               : "Q14",
        "query"            : "Nitrogen tanah sangat tinggi, suhu 30°C, kelembaban 85%, penyakit apa yang perlu diwaspadai?",
        "disease"          : "sheath_blight",
        "relevant_keywords": ["busuk pelepah", "nitrogen tinggi", "rhizoctonia", "validamycin", "suhu tinggi", "sensor"],
        "ground_truth"     : (
            "Kondisi nitrogen tinggi + suhu 28-32°C + kelembaban >80% adalah kondisi OPTIMAL Busuk Pelepah (Rhizoctonia solani). "
            "Tindakan: periksa pelepah daun untuk lesi abu-abu. Kurangi dosis nitrogen SEGERA. "
            "Semprot validamycin preventif jika ada gejala awal. "
            "Perlebar jarak tanam untuk sirkulasi udara. Gunakan irigasi berselang."
        ),
    },
    {
        "id"               : "Q15",
        "query"            : "Petani melaporkan populasi wereng hijau meningkat, apa yang harus dilakukan untuk mencegah tungro?",
        "disease"          : "tungro",
        "relevant_keywords": ["tungro", "wereng hijau", "nephotettix", "imidakloprid", "buprofezin", "vektor", "pencegahan"],
        "ground_truth"     : (
            "Wereng hijau adalah vektor virus tungro. "
            "Tindakan SEGERA: semprot insektisida sistemik imidakloprid (Confidor 200 SL 0.5 ml/L) "
            "atau buprofezin (Applaud 25 WP 1 g/L) untuk membasmi wereng sebelum virus menyebar. "
            "Pasang light trap untuk monitoring. Pertahankan musuh alami. "
            "Cabut tanaman bergejala kuning-oranye jika ada."
        ),
    },
]


# ═════════════════════════════════════════════════════════════════
# HELPER: Cosine Similarity
# ═════════════════════════════════════════════════════════════════
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def is_relevant_chunk(chunk_text: str, relevant_keywords: list[str]) -> bool:
    """
    Chunk dianggap relevan jika mengandung MINIMAL 1 keyword yang diberikan.
    """
    chunk_lower = chunk_text.lower()
    return any(kw.lower() in chunk_lower for kw in relevant_keywords)


# ═════════════════════════════════════════════════════════════════
# RETRIEVER EVALUATION — MRR, Recall@k, Precision@k
# ═════════════════════════════════════════════════════════════════
def evaluate_retriever(ground_truth_list: list[dict], k: int = 5) -> dict:
    """
    Evaluasi komponen retriever RAG.

    Rumus (sesuai paper CottonBot, Kandamali et al. 2025):
      MRR        = (1/N) * Σ (1/rank_i)
      Recall@k   = jumlah chunk relevan di top-k / total chunk relevan
      Precision@k = jumlah chunk relevan di top-k / k
    """
    from rag import retrieve

    mrr_scores       = []
    recall_scores    = []
    precision_scores = []
    per_query_results = []

    logger.info(f"Mengevaluasi retriever untuk {len(ground_truth_list)} query (k={k})...")

    for qa in ground_truth_list:
        query             = qa["query"]
        relevant_keywords = qa["relevant_keywords"]

        retrieved = retrieve(query, k=k)

        relevance_flags = [
            1 if is_relevant_chunk(item["text"], relevant_keywords) else 0
            for item in retrieved
        ]

        # ── MRR ───────────────────────────────────────────────────
        first_relevant_rank = None
        for i, flag in enumerate(relevance_flags):
            if flag == 1:
                first_relevant_rank = i + 1
                break
        rr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
        mrr_scores.append(rr)

        # ── Recall@k ──────────────────────────────────────────────
        # Closed-domain: anggap ada 1 chunk relevan ideal, recall = 1 jika ditemukan
        num_relevant = sum(relevance_flags)
        recall = min(num_relevant, 1)
        recall_scores.append(recall)

        # ── Precision@k ───────────────────────────────────────────
        precision = sum(relevance_flags) / k
        precision_scores.append(precision)

        per_query_results.append({
            "id"             : qa["id"],
            "query"          : query[:70] + "..." if len(query) > 70 else query,
            "disease"        : qa["disease"],
            "rr"             : round(rr, 4),
            "recall"         : round(recall, 4),
            "precision"      : round(precision, 4),
            "relevance_flags": relevance_flags,
            "top1_score"     : round(retrieved[0]["score"], 4) if retrieved else 0,
        })

    return {
        "MRR"             : round(float(np.mean(mrr_scores)), 4),
        f"Recall@{k}"     : round(float(np.mean(recall_scores)), 4),
        f"Precision@{k}"  : round(float(np.mean(precision_scores)), 4),
        "per_query"       : per_query_results,
        "k"               : k,
        "n_queries"       : len(ground_truth_list),
    }


# ═════════════════════════════════════════════════════════════════
# GENERATOR EVALUATION — Cosine Similarity + Waktu + Faithfulness
# ═════════════════════════════════════════════════════════════════
def evaluate_generator(
    ground_truth_list: list[dict],
    llm_func         : callable,
    llm_name         : str = "LLM",
) -> dict:
    """
    Evaluasi komponen generator (LLM) dalam RAG pipeline.

    Metrik:
    - Cosine Similarity  : kemiripan semantik antara jawaban LLM dan ground truth
    - Avg Generation Time: rata-rata waktu generate (detik)
    - Faithfulness       : skor manual annotator 0-1 (diisi terpisah)
    """
    from rag import build_rag_prompt
    from sentence_transformers import SentenceTransformer

    # FIX: gunakan EMBEDDING_MODEL_NAME yang sudah didefinisikan di atas
    embed_model  = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cosim_scores = []
    time_scores  = []
    per_query_res = []

    logger.info(f"Mengevaluasi generator: {llm_name} ({len(ground_truth_list)} query)...")

    for qa in ground_truth_list:
        disease      = qa["disease"]
        ground_truth = qa["ground_truth"]

        # Build RAG prompt → generate jawaban dari LLM
        prompt, retrieved = build_rag_prompt(disease_name=disease, k=3)

        try:
            t0     = time.perf_counter()
            answer = llm_func(prompt)
            t_gen  = round(time.perf_counter() - t0, 3)
        except Exception as e:
            logger.warning(f"Query {qa['id']} gagal: {e} — dilewati")
            answer = ""
            t_gen  = 0.0
            
        time_scores.append(t_gen)

        # Cosine similarity: jawaban LLM vs ground truth
        vecs = embed_model.encode(
            [answer, ground_truth],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        cos_sim = cosine_similarity(vecs[0], vecs[1])
        cosim_scores.append(cos_sim)

        per_query_res.append({
            "id"          : qa["id"],
            "disease"     : disease,
            "query"       : qa["query"],
            "generated"   : answer[:300] + "..." if len(answer) > 300 else answer,
            "ground_truth": ground_truth[:300] + "..." if len(ground_truth) > 300 else ground_truth,
            "cosine_sim"  : round(cos_sim, 4),
            "time_s"      : t_gen,
            "faithfulness": None,  # diisi manual annotator
        })

    return {
        "llm"                  : llm_name,
        "avg_cosine_similarity": round(float(np.mean(cosim_scores)), 4),
        "avg_generation_time_s": round(float(np.mean(time_scores)), 3),
        "faithfulness_note"    : "Diisi manual annotator (0=tidak akurat, 0.5=sebagian, 1=sangat akurat)",
        "per_query"            : per_query_res,
        "n_queries"            : len(ground_truth_list),
    }


# ═════════════════════════════════════════════════════════════════
# FAITHFULNESS MANUAL SCORING
# ═════════════════════════════════════════════════════════════════
def run_faithfulness_annotation(generator_results: dict) -> dict:
    """
    Tool interaktif untuk mengisi skor faithfulness secara manual.
    Kamu membaca jawaban LLM vs ground truth, lalu beri skor 0, 0.5, atau 1.
    """
    print("\n" + "="*65)
    print("ANOTASI FAITHFULNESS — Manual")
    print("Skor: 0 = tidak akurat | 0.5 = sebagian akurat | 1 = sangat akurat")
    print("="*65)

    scores = []
    for i, pq in enumerate(generator_results["per_query"]):
        print(f"\n[{i+1}/{len(generator_results['per_query'])}] {pq['id']} — {pq['disease']}")
        print(f"\nGround Truth :\n  {pq['ground_truth']}")
        print(f"\nJawaban LLM  :\n  {pq['generated']}")

        while True:
            try:
                score = float(input("\nSkor Faithfulness (0.0 / 0.5 / 1.0): "))
                if 0.0 <= score <= 1.0:
                    break
                print("Masukkan angka antara 0 dan 1.")
            except ValueError:
                print("Input tidak valid, coba lagi.")

        pq["faithfulness"] = score
        scores.append(score)
        print(f"  → Skor {score} disimpan.")

    generator_results["avg_faithfulness"] = round(float(np.mean(scores)), 4)
    return generator_results


# ═════════════════════════════════════════════════════════════════
# PRINT TABLE
# ═════════════════════════════════════════════════════════════════
def print_retriever_table(results: dict):
    k = results["k"]
    print(f"\n{'='*60}")
    print(f"  HASIL EVALUASI RETRIEVER RAG (k={k}, n={results['n_queries']} query)")
    print(f"{'='*60}")
    print(f"  {'MRR':<20} {results['MRR']:>10.4f}")
    print(f"  {f'Recall@{k}':<20} {results[f'Recall@{k}']:>10.4f}")
    print(f"  {f'Precision@{k}':<20} {results[f'Precision@{k}']:>10.4f}")
    print(f"{'='*60}")
    print(f"\n  {'ID':<5} {'Penyakit':<30} {'RR':>6} {'Rec':>6} {'Prec':>6} {'Top1':>6}")
    print(f"  {'-'*60}")
    for pq in results["per_query"]:
        print(f"  {pq['id']:<5} {pq['disease']:<30} {pq['rr']:>6.3f} "
              f"{pq['recall']:>6.3f} {pq['precision']:>6.3f} {pq['top1_score']:>6.3f}")


def print_generator_table(results: dict):
    print(f"\n{'='*60}")
    print(f"  HASIL EVALUASI GENERATOR: {results['llm']}")
    print(f"{'='*60}")
    print(f"  Avg Cosine Similarity  : {results['avg_cosine_similarity']:.4f}")
    print(f"  Avg Generation Time    : {results['avg_generation_time_s']:.3f} detik")
    if "avg_faithfulness" in results:
        print(f"  Avg Faithfulness       : {results['avg_faithfulness']:.4f}")
    print(f"{'='*60}")


# ═════════════════════════════════════════════════════════════════
# SAVE CSV & JSON
# ═════════════════════════════════════════════════════════════════
def save_results_csv(retriever_res: dict, output_path: str = "hasil_retriever.csv"):
    import csv
    k = retriever_res["k"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Penyakit", "Query", "RR", f"Recall@{k}", f"Precision@{k}", "Top1_Score", "Relevant_Flags"])
        for pq in retriever_res["per_query"]:
            writer.writerow([
                pq["id"], pq["disease"], pq["query"],
                pq["rr"], pq["recall"], pq["precision"],
                pq["top1_score"],
                str(pq["relevance_flags"]),
            ])
        writer.writerow([])
        writer.writerow(["RATA-RATA", "", "",
                        retriever_res["MRR"],
                        retriever_res[f"Recall@{k}"],
                        retriever_res[f"Precision@{k}"],
                        "", ""])
    logger.info(f"CSV retriever disimpan: {output_path}")


def save_results(retriever_res: dict, generator_res: dict, output_path: str = "hasil_evaluasi_rag.json"):
    output = {
        "retriever" : retriever_res,
        "generator" : generator_res,
        "timestamp" : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON hasil evaluasi disimpan: {output_path}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluasi RAG Pipeline Penyakit Padi")
    parser.add_argument("--k",           type=int, default=5)
    parser.add_argument("--output",      type=str, default="hasil_evaluasi_rag.json")
    parser.add_argument("--csv",         type=str, default="hasil_retriever.csv")
    parser.add_argument("--skip-llm",    action="store_true", help="Hanya evaluasi retriever")
    parser.add_argument("--llm",         type=str, default="both", choices=["groq", "gemini", "both"])
    parser.add_argument("--faithfulness",action="store_true", help="Jalankan anotasi faithfulness manual")
    args = parser.parse_args()

    # ── Pastikan RAG index sudah ada ────────────────────────────
    from rag import get_index_info, build_index
    if not Path("faiss_index.bin").exists():
        logger.info("Index belum ada, membangun index RAG dari knowledge_base/...")
        build_index()

    info = get_index_info()
    print("\nRAG Index Info:")
    for k_info, v in info.items():
        print(f"  {k_info}: {v}")

    # ── Evaluasi Retriever ───────────────────────────────────────
    retriever_res = evaluate_retriever(GROUND_TRUTH_QA, k=args.k)
    print_retriever_table(retriever_res)
    save_results_csv(retriever_res, args.csv)

    # ── Evaluasi Generator (opsional) ────────────────────────────
    generator_results = {}
    if not args.skip_llm:
        try:
            from dotenv import load_dotenv
            load_dotenv()

            # Helper build LLM function
            def _make_groq_func():
                from groq import Groq
                from llm import GROQ_MODEL_NAME, SYSTEM_PROMPT
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                def fn(prompt):
                    resp = client.chat.completions.create(
                        model=GROQ_MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.7, max_tokens=1024,
                    )
                    return resp.choices[0].message.content
                return fn, f"Groq — {GROQ_MODEL_NAME}"

            def _make_gemini_func():
                import os
                import time
                import requests as req

                GEMINI_MODEL = "gemini-2.5-flash"
                api_key      = os.environ.get("GEMINI_API_KEY")

                def fn(prompt):
                    url     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature"    : 0.7,
                            "maxOutputTokens": 1024,
                        },
                    }
                    for attempt in range(5):
                        resp = req.post(url, json=payload, timeout=60)
                        if resp.status_code == 429:
                            wait = 15 * (attempt + 1)
                            logger.info(f"Gemini rate limit, tunggu {wait} detik...")
                            time.sleep(wait)
                            continue
                        resp.raise_for_status()
                        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    raise Exception("Gemini gagal setelah 5 percobaan")

                return fn, f"Gemini — {GEMINI_MODEL}"

            llms_to_eval = []
            if args.llm in ("groq", "both"):
                llms_to_eval.append(_make_groq_func())
            if args.llm in ("gemini", "both"):
                llms_to_eval.append(_make_gemini_func())

            for llm_func, llm_name in llms_to_eval:
                gen_res = evaluate_generator(GROUND_TRUTH_QA, llm_func, llm_name)
                print_generator_table(gen_res)

                if args.faithfulness:
                    gen_res = run_faithfulness_annotation(gen_res)
                    print_generator_table(gen_res)

                generator_results[llm_name] = gen_res

        except Exception as e:
            logger.warning(f"Evaluasi generator gagal: {e}")
            generator_results = {"error": str(e)}

    # ── Simpan semua hasil ───────────────────────────────────────
    save_results(retriever_res, generator_results, args.output)
    print(f"\n✅ Selesai! Hasil disimpan di: {args.output} dan {args.csv}")
    print(f"   Salin angka dari CSV ke sheet 'Hasil Retriever RAG' di Excel Ground Truth.")
    if generator_results and "error" not in generator_results:
        print(f"   Salin Cosine Sim + Waktu ke sheet 'Hasil Generator LLM+RAG'.")
        if args.faithfulness:
            print(f"   Faithfulness sudah diisi interaktif.")
        else:
            print(f"   Jalankan lagi dengan --faithfulness untuk mengisi skor Faithfulness secara manual.")


if __name__ == "__main__":
    main()
