"""
evaluate_rag.py — Evaluasi RAG Pipeline Penyakit Padi
======================================================
Menghitung metrik evaluasi sesuai arahan dosen:

RETRIEVER METRICS:
  - MRR (Mean Reciprocal Rank)
  - Hit@k  (binary: ditemukan minimal 1 chunk relevan di top-k)
  - Precision@k

GENERATOR METRICS:
  - ROUGE-1, ROUGE-2, ROUGE-L  — evaluasi kualitas teks output LLM
  - Cosine Similarity (semantic similarity vs ground truth)
  - Average Generation Time (detik)
  - Faithfulness (manual scoring 0–1, opsional via --faithfulness)

3 TIER MODEL LLM YANG DIEVALUASI:
  - LOW    : Qwen2.5-3B        (Ollama, lokal) — model kecil, sangat cepat
  - MEDIUM : Gemini 2.5 Flash  (Google API)    — seimbang, context 1M token
  - HIGH   : Llama3.3-70B      (Groq API)      — model besar, kualitas tinggi, cepat via cloud

Cara pakai:
  python evaluate_rag.py --k 3 --llm all    --output hasil_evaluasi_rag.json
  python evaluate_rag.py --k 5 --llm low
  python evaluate_rag.py --k 5 --llm medium
  python evaluate_rag.py --k 5 --llm high
  python evaluate_rag.py --k 5 --skip-llm                   (hanya retriever)
  python evaluate_rag.py --k 3 --llm all --faithfulness     (+ anotasi manual)

Changelog:
  v5 (Jun 2026): Ganti HIGH=Llama3.3-70B dari Ollama ke Groq API
  v4 (Jun 2026): Ganti LOW=Qwen2.5-3B(Ollama), MEDIUM=Gemini2.5Flash, HIGH=Llama3.3-70B(Ollama)
                 Hapus dependensi Groq API, tambah Ollama factory
  v3 (Jun 2026): +ROUGE-1/2/L metrics, 3 tier LLM (LOW/MEDIUM/LARGE)
  v2 (Jun 2025): +Q16 bacterial_leaf_streak, +Q17-Q18 harvest_stage
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
# KONFIGURASI
# ═════════════════════════════════════════════════════════════════
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 3 Tier Model LLM
LLM_LOW_MODEL    = "qwen2.5:3b"                 # LOW    — Ollama lokal
LLM_MEDIUM_MODEL = "gemini-2.5-flash"           # MEDIUM — Google API
LLM_HIGH_MODEL   = "llama-3.3-70b-versatile"   # HIGH   — Groq API

# Fallback list Gemini (MEDIUM)
GEMINI_CANDIDATES = [
    ("gemini-2.5-flash",      "v1beta"),
    ("gemini-2.5-flash",      "v1"),
    ("gemini-2.0-flash",      "v1"),
    ("gemini-2.0-flash",      "v1beta"),
    ("gemini-2.0-flash-lite", "v1"),
]

# System prompt lokal (aman tanpa import llm.py)
_SYSTEM_PROMPT = (
    "Kamu adalah asisten pertanian yang membantu petani padi langsung di lapangan. "
    "Gunakan bahasa sederhana dan mudah dipahami. "
    "Jawab langsung ke solusi yang bisa dikerjakan hari ini. "
    "Gunakan informasi dari basis pengetahuan sebagai acuan utama. "
    "Selalu jawab dalam Bahasa Indonesia."
)


# ═════════════════════════════════════════════════════════════════
# GROUND TRUTH — 18 Skenario
# ═════════════════════════════════════════════════════════════════
GROUND_TRUTH_QA = [
    # ── Q01 ── bacterial_leaf_blight ──────────────────────────────
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
    # ── Q02 ── leaf_blast ─────────────────────────────────────────
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
    # ── Q03 ── neck_blast ─────────────────────────────────────────
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
    # ── Q04 ── sheath_blight ──────────────────────────────────────
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
    # ── Q05 ── tungro ─────────────────────────────────────────────
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
    # ── Q06 ── brown_spot ─────────────────────────────────────────
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
    # ── Q07 ── dead_heart ─────────────────────────────────────────
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
    # ── Q08 ── hispa ──────────────────────────────────────────────
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
    # ── Q09 ── bacterial_panicle_blight ───────────────────────────
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
    # ── Q10 ── leaf_smut ──────────────────────────────────────────
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
    # ── Q11 ── downy_mildew ───────────────────────────────────────
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
    # ── Q12 ── healthy ────────────────────────────────────────────
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
    # ── Q13 ── leaf_blast (sensor) ────────────────────────────────
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
    # ── Q14 ── sheath_blight (sensor) ────────────────────────────
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
    # ── Q15 ── tungro (pencegahan vektor) ─────────────────────────
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
    # ── Q16 ── bacterial_leaf_streak ──────────────────────────────
    {
        "id"               : "Q16",
        "query"            : "Ada garis-garis sempit kuning di antara tulang daun padi disertai butiran eksudat kekuningan, penyakit apa ini?",
        "disease"          : "bacterial_leaf_streak",
        "relevant_keywords": ["hawar daun bergaris", "xanthomonas oryzicola", "bismerthiazol", "garis sempit", "interveinal", "eksudat"],
        "ground_truth"     : (
            "Hawar Daun Bergaris Bakteri disebabkan Xanthomonas oryzae pv. oryzicola. "
            "Gejala: lesi garis-garis sempit kuning di antara tulang daun (interveinal) dengan butiran eksudat kekuningan. "
            "Penanganan: semprot bakterisida bismerthiazol (Xanthocide 20 WP 2 g/L) atau copper-based (Cupravit OB 21 2-3 g/L), "
            "hindari nitrogen berlebih, perbaiki sirkulasi udara. "
            "Pencegahan: perlakuan benih, jarak tanam optimal, rotasi tanaman."
        ),
    },
    # ── Q17 ── harvest_stage ──────────────────────────────────────
    {
        "id"               : "Q17",
        "query"            : "Malai padi sudah menunduk dan 80% gabah berwarna kuning keemasan, apakah sudah siap panen dan apa langkahnya?",
        "disease"          : "harvest_stage",
        "relevant_keywords": ["fase panen", "kadar air", "gabah", "panen", "malai menunduk", "perontokan", "pengeringan"],
        "ground_truth"     : (
            "Padi yang 80-95% gabahnya sudah kuning keemasan dan malai menunduk menandakan fase panen optimal. "
            "Ukur kadar air gabah — idealnya 20-25% untuk panen. "
            "Jadwalkan panen dalam 3-5 hari ke depan saat cuaca cerah. "
            "Lakukan panen pagi/sore hari, rontokkan gabah dalam 24 jam. "
            "Keringkan hingga kadar air ≤14% sebelum disimpan di gudang yang kering."
        ),
    },
    # ── Q18 ── harvest_stage (sensor) ─────────────────────────────
    {
        "id"               : "Q18",
        "query"            : "Kelembaban udara 85% saat padi siap panen, apa risiko kualitas gabah dan apa yang harus dilakukan?",
        "disease"          : "harvest_stage",
        "relevant_keywords": ["panen", "kelembaban", "pengeringan", "busuk gabah", "aflatoksin", "mechanical dryer"],
        "ground_truth"     : (
            "Kelembaban udara >80% saat panen meningkatkan risiko busuk gabah (Fusarium, Burkholderia glumae) "
            "dan membuat gabah sulit kering secara alami. "
            "Gunakan pengeringan mekanis (mechanical dryer 40-43°C) jika cuaca tidak mendukung penjemuran. "
            "Rontokkan gabah dalam 24 jam setelah panen. "
            "Pastikan kadar air gabah turun ke ≤14% sebelum disimpan untuk mencegah aflatoksin."
        ),
    },
]


# ═════════════════════════════════════════════════════════════════
# HELPER: Cosine Similarity
# ═════════════════════════════════════════════════════════════════
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot = float(np.dot(vec_a, vec_b))
    return max(-1.0, min(1.0, dot))


# ═════════════════════════════════════════════════════════════════
# ROUGE Scoring
# ═════════════════════════════════════════════════════════════════
def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Hitung skor ROUGE antara output LLM dan ground truth.
    ROUGE-1: overlap unigram | ROUGE-2: bigram | ROUGE-L: LCS
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": {"precision": round(scores["rouge1"].precision, 4),
                       "recall"   : round(scores["rouge1"].recall,    4),
                       "fmeasure" : round(scores["rouge1"].fmeasure,  4)},
            "rouge2": {"precision": round(scores["rouge2"].precision, 4),
                       "recall"   : round(scores["rouge2"].recall,    4),
                       "fmeasure" : round(scores["rouge2"].fmeasure,  4)},
            "rougeL": {"precision": round(scores["rougeL"].precision, 4),
                       "recall"   : round(scores["rougeL"].recall,    4),
                       "fmeasure" : round(scores["rougeL"].fmeasure,  4)},
        }
    except ImportError:
        logger.error("rouge-score belum terinstall. Jalankan: pip install rouge-score")
        return {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        }


# ═════════════════════════════════════════════════════════════════
# RETRIEVER EVALUATION
# ═════════════════════════════════════════════════════════════════
def evaluate_retriever(ground_truth_list: list[dict], k: int = 5) -> dict:
    from rag import retrieve

    mrr_scores  = []
    hit_scores  = []
    prec_scores = []
    per_query   = []

    logger.info(f"Mengevaluasi retriever untuk {len(ground_truth_list)} query (k={k})...")

    for qa in ground_truth_list:
        query    = qa["query"]
        keywords = [kw.lower() for kw in qa["relevant_keywords"]]

        # retrieve() di rag.py sudah handle encoding sendiri — kirim string langsung
        chunks = retrieve(query, k=k)

        flags = []
        for chunk in chunks:
            text_lower = chunk["text"].lower()
            relevant   = any(kw in text_lower for kw in keywords)
            flags.append(1 if relevant else 0)

        rr = 0.0
        for rank, flag in enumerate(flags, start=1):
            if flag == 1:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

        hit  = 1.0 if any(flags) else 0.0
        prec = sum(flags) / k
        hit_scores.append(hit)
        prec_scores.append(prec)

        per_query.append({
            "id"             : qa["id"],
            "disease"        : qa["disease"],
            "query"          : query,
            "rr"             : round(rr,   4),
            "hit_k"          : hit,
            "precision_k"    : round(prec, 4),
            "top1_score"     : round(chunks[0]["score"], 4) if chunks else 0.0,
            "relevance_flags": flags,
        })

    return {
        "MRR"             : round(float(np.mean(mrr_scores)),  4),
        f"Hit@{k}"        : round(float(np.mean(hit_scores)),  4),
        f"Precision@{k}"  : round(float(np.mean(prec_scores)), 4),
        "k"               : k,
        "n_queries"       : len(ground_truth_list),
        "per_query"       : per_query,
    }


# ═════════════════════════════════════════════════════════════════
# FACTORY: Ollama (LOW)
# ═════════════════════════════════════════════════════════════════
def _make_ollama_func(model_name: str, label: str):
    """
    Buat generator Ollama untuk model lokal (Qwen 3B / Llama 70B).
    Timeout 600s untuk mengakomodasi model 70B yang lambat.
    Pastikan Ollama sudah running dan model sudah di-pull.
    """
    import requests as _req

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    # Test koneksi Ollama saat inisialisasi
    try:
        r = _req.get(f"{base_url}/api/tags", timeout=5)
        if r.status_code != 200:
            raise ConnectionError(f"Ollama merespons {r.status_code}")
        # Cek model tersedia
        available = [m["name"] for m in r.json().get("models", [])]
        if model_name not in available:
            logger.warning(
                f"Model '{model_name}' belum ada di Ollama. "
                f"Jalankan: ollama pull {model_name}"
            )
    except _req.exceptions.ConnectionError:
        raise ConnectionError(
            f"Ollama tidak bisa diakses di {base_url}.\n"
            f"Pastikan Ollama sudah berjalan: ollama serve"
        )

    def fn(prompt: str) -> str:
        payload = {
            "model"   : model_name,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream" : False,
            "options": {"temperature": 0.7, "num_predict": 1024},
        }
        r = _req.post(f"{base_url}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    return fn, label


# ═════════════════════════════════════════════════════════════════
# FACTORY: Gemini (MEDIUM)
# ═════════════════════════════════════════════════════════════════
def _make_gemini_func():
    """Buat generator Gemini 2.5 Flash (MEDIUM) dengan fallback model list."""
    import requests as _req

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY belum diset di .env")

    _base   = "https://generativelanguage.googleapis.com"
    _active = [None]

    def fn(prompt: str) -> str:
        ordered = [_active[0]] + [c for c in GEMINI_CANDIDATES if c != _active[0]] \
                  if _active[0] else list(GEMINI_CANDIDATES)

        payload = {
            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents"         : [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig" : {"temperature": 0.7, "maxOutputTokens": 1024},
        }

        last_err = None
        for model, ver in ordered:
            url = f"{_base}/{ver}/models/{model}:generateContent?key={api_key}"
            for attempt in range(3):
                resp = _req.post(url, json=payload, timeout=60)
                if resp.status_code == 429:
                    wait = 15 * (attempt + 1)
                    logger.info(f"Gemini rate limit ({model}), tunggu {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 200:
                    parts = (
                        resp.json()
                        .get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [])
                    )
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if text:
                        _active[0] = (model, ver)
                        return text
                last_err = f"HTTP {resp.status_code}"
                break

        raise RuntimeError(f"Semua model Gemini gagal. Error terakhir: {last_err}")

    active_name = GEMINI_CANDIDATES[0][0]
    return fn, f"Gemini — {active_name} (MEDIUM)"


# ═════════════════════════════════════════════════════════════════
# FACTORY: Groq (HIGH)
# ═════════════════════════════════════════════════════════════════
def _make_groq_func():
    """
    Buat generator Groq untuk Llama 3.3-70B (HIGH).
    Menggunakan OpenAI-compatible endpoint Groq.
    Sangat cepat — Groq menggunakan LPU (Language Processing Unit).
    """
    import requests as _req

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY belum diset di .env")

    _groq_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type" : "application/json",
    }

    def fn(prompt: str) -> str:
        payload = {
            "model"      : LLM_HIGH_MODEL,
            "messages"   : [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens" : 1024,
        }
        for attempt in range(3):
            try:
                r = _req.post(_groq_url, headers=headers, json=payload, timeout=60)
                if r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    logger.info(f"Groq rate limit, tunggu {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except _req.exceptions.Timeout:
                raise RuntimeError(f"Groq timeout (>60s) untuk model {LLM_HIGH_MODEL}")
        raise RuntimeError("Groq gagal setelah 3 percobaan (rate limit)")

    return fn, f"Groq — {LLM_HIGH_MODEL} (HIGH)"
def evaluate_generator(
    ground_truth_list: list[dict],
    llm_func         : callable,
    llm_name         : str = "LLM",
    k                : int = 5,
) -> dict:
    from rag import build_rag_prompt
    from sentence_transformers import SentenceTransformer

    embed_model   = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cosim_scores  = []
    time_scores   = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    per_query_res = []

    logger.info(f"Mengevaluasi generator: {llm_name} ({len(ground_truth_list)} query, k={k})...")

    for qa in ground_truth_list:
        disease      = qa["disease"]
        ground_truth = qa["ground_truth"]
        prompt, _    = build_rag_prompt(disease_name=disease, k=k)

        try:
            t0     = time.perf_counter()
            answer = llm_func(prompt)
            t_gen  = round(time.perf_counter() - t0, 3)
        except Exception as e:
            logger.warning(f"Query {qa['id']} gagal: {e} — dilewati")
            answer = ""
            t_gen  = 0.0

        time_scores.append(t_gen)

        if answer:
            vecs = embed_model.encode(
                [answer, ground_truth],
                convert_to_numpy=True, normalize_embeddings=True,
            )
            cos_sim = cosine_similarity(vecs[0], vecs[1])
        else:
            cos_sim = 0.0
        cosim_scores.append(cos_sim)

        rouge = compute_rouge(answer, ground_truth) if answer else {
            "rouge1": {"fmeasure": 0.0},
            "rouge2": {"fmeasure": 0.0},
            "rougeL": {"fmeasure": 0.0},
        }
        rouge1_scores.append(rouge["rouge1"]["fmeasure"])
        rouge2_scores.append(rouge["rouge2"]["fmeasure"])
        rougeL_scores.append(rouge["rougeL"]["fmeasure"])

        per_query_res.append({
            "id"          : qa["id"],
            "disease"     : disease,
            "query"       : qa["query"],
            "generated"   : answer[:300] + "..." if len(answer) > 300 else answer,
            "ground_truth": ground_truth[:1000] + "..." if len(ground_truth) > 1000 else ground_truth,
            "cosine_sim"  : round(cos_sim, 4),
            "rouge1_f"    : rouge["rouge1"]["fmeasure"],
            "rouge2_f"    : rouge["rouge2"]["fmeasure"],
            "rougeL_f"    : rouge["rougeL"]["fmeasure"],
            "rouge_detail": rouge,
            "time_s"      : t_gen,
            "faithfulness": None,
        })

    return {
        "llm"                  : llm_name,
        "avg_cosine_similarity": round(float(np.mean(cosim_scores)),  4),
        "avg_rouge1_f"         : round(float(np.mean(rouge1_scores)), 4),
        "avg_rouge2_f"         : round(float(np.mean(rouge2_scores)), 4),
        "avg_rougeL_f"         : round(float(np.mean(rougeL_scores)), 4),
        "avg_generation_time_s": round(float(np.mean(time_scores)),   3),
        "faithfulness_note"    : "Diisi manual annotator (0=tidak akurat, 0.5=sebagian, 1=sangat akurat)",
        "per_query"            : per_query_res,
        "n_queries"            : len(ground_truth_list),
        "k_used"               : k,
    }


# ═════════════════════════════════════════════════════════════════
# FAITHFULNESS MANUAL SCORING
# ═════════════════════════════════════════════════════════════════
def run_faithfulness_annotation(generator_results: dict) -> dict:
    print("\n" + "="*65)
    print(f"ANOTASI FAITHFULNESS — {generator_results['llm']}")
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
    print(f"\n{'='*65}")
    print(f"  HASIL EVALUASI RETRIEVER RAG (k={k}, n={results['n_queries']} query)")
    print(f"{'='*65}")
    print(f"  {'MRR':<22} {results['MRR']:>10.4f}")
    print(f"  {f'Hit@{k}':<22} {results[f'Hit@{k}']:>10.4f}")
    print(f"  {f'Precision@{k}':<22} {results[f'Precision@{k}']:>10.4f}")
    print(f"{'='*65}")
    print(f"\n  {'ID':<5} {'Penyakit':<30} {'RR':>6} {'Hit':>5} {'Prec':>6}")
    print(f"  {'-'*55}")
    for pq in results["per_query"]:
        print(
            f"  {pq['id']:<5} {pq['disease']:<30} {pq['rr']:>6.3f} "
            f"{pq['hit_k']:>5.1f} {pq['precision_k']:>6.3f}"
        )


def print_generator_table(results: dict):
    print(f"\n{'='*65}")
    print(f"  HASIL EVALUASI GENERATOR: {results['llm']}")
    print(f"{'='*65}")
    print(f"  Avg ROUGE-1 F1         : {results['avg_rouge1_f']:.4f}")
    print(f"  Avg ROUGE-2 F1         : {results['avg_rouge2_f']:.4f}")
    print(f"  Avg ROUGE-L F1         : {results['avg_rougeL_f']:.4f}")
    print(f"  Avg Cosine Similarity  : {results['avg_cosine_similarity']:.4f}")
    print(f"  Avg Generation Time    : {results['avg_generation_time_s']:.3f} detik")
    print(f"  k chunks dipakai       : {results.get('k_used', 'N/A')}")
    if "avg_faithfulness" in results:
        print(f"  Avg Faithfulness       : {results['avg_faithfulness']:.4f}")
    print(f"{'='*65}")
    print(f"\n  {'ID':<5} {'Penyakit':<28} {'R1':>6} {'R2':>6} {'RL':>6} {'CosSim':>8} {'t(s)':>6}")
    print(f"  {'-'*68}")
    for pq in results["per_query"]:
        print(
            f"  {pq['id']:<5} {pq['disease']:<28} "
            f"{pq['rouge1_f']:>6.3f} {pq['rouge2_f']:>6.3f} {pq['rougeL_f']:>6.3f} "
            f"{pq['cosine_sim']:>8.4f} {pq['time_s']:>6.2f}"
        )


# ═════════════════════════════════════════════════════════════════
# SAVE CSV & JSON
# ═════════════════════════════════════════════════════════════════
def save_retriever_csv(retriever_res: dict, output_path: str = "hasil_retriever.csv"):
    import csv
    k = retriever_res["k"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Penyakit", "Query", "RR", f"Hit@{k}", f"Precision@{k}", "Top1_Score", "Relevant_Flags"])
        for pq in retriever_res["per_query"]:
            writer.writerow([pq["id"], pq["disease"], pq["query"], pq["rr"],
                             pq["hit_k"], pq["precision_k"], pq["top1_score"],
                             str(pq["relevance_flags"])])
        writer.writerow([])
        writer.writerow(["RATA-RATA", "", "", retriever_res["MRR"],
                         retriever_res[f"Hit@{k}"], retriever_res[f"Precision@{k}"], "", ""])
    logger.info(f"CSV retriever disimpan: {output_path}")


def save_generator_csv(generator_results: dict, output_path: str = "hasil_generator.csv"):
    import csv
    rows = []
    for llm_name, gen_res in generator_results.items():
        if "error" in str(gen_res):
            continue
        for pq in gen_res.get("per_query", []):
            rows.append({
                "LLM"         : llm_name,
                "ID"          : pq["id"],
                "Penyakit"    : pq["disease"],
                "ROUGE-1 F1"  : pq.get("rouge1_f", ""),
                "ROUGE-2 F1"  : pq.get("rouge2_f", ""),
                "ROUGE-L F1"  : pq.get("rougeL_f", ""),
                "Cosine Sim"  : pq.get("cosine_sim", ""),
                "Waktu (s)"   : pq.get("time_s", ""),
                "Faithfulness": pq.get("faithfulness", ""),
            })
    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["=== RATA-RATA PER LLM ==="])
            writer.writerow(["LLM", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1",
                             "Cosine Sim", "Avg Waktu (s)", "Avg Faithfulness"])
            for llm_name, gen_res in generator_results.items():
                if "error" in str(gen_res):
                    continue
                writer.writerow([
                    llm_name,
                    gen_res.get("avg_rouge1_f", ""),
                    gen_res.get("avg_rouge2_f", ""),
                    gen_res.get("avg_rougeL_f", ""),
                    gen_res.get("avg_cosine_similarity", ""),
                    gen_res.get("avg_generation_time_s", ""),
                    gen_res.get("avg_faithfulness", ""),
                ])
    logger.info(f"CSV generator disimpan: {output_path}")


def save_results(
    retriever_res    : dict,
    generator_results: dict,
    output_path      : str = "hasil_evaluasi_rag.json",
):
    output = {
        "metadata": {
            "timestamp"      : time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_queries"      : retriever_res["n_queries"],
            "k"              : retriever_res["k"],
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_models": {
                "low"   : f"Ollama — {LLM_LOW_MODEL} (lokal)",
                "medium": f"Google — {LLM_MEDIUM_MODEL} (API)",
                "high"  : f"Groq   — {LLM_HIGH_MODEL} (API)",
            },
        },
        "retriever" : retriever_res,
        "generator" : generator_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON hasil evaluasi disimpan: {output_path}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi RAG Pipeline Penyakit Padi — 3 Tier LLM + ROUGE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python evaluate_rag.py --k 3 --llm all    --output hasil_evaluasi_rag.json
  python evaluate_rag.py --k 5 --llm low
  python evaluate_rag.py --k 5 --llm medium
  python evaluate_rag.py --k 5 --llm high
  python evaluate_rag.py --k 5 --skip-llm
  python evaluate_rag.py --k 3 --llm all --faithfulness

Penjelasan tier LLM:
  low    → Qwen2.5-3B              (Ollama lokal) — model kecil, sangat cepat
  medium → Gemini 2.5 Flash        (Google API)   — seimbang, context 1M token
  high   → Llama3.3-70B-versatile  (Groq API)     — model besar, cepat via cloud
  all    → evaluasi ketiga tier sekaligus

Prasyarat:
  Ollama (untuk LOW): ollama serve && ollama pull qwen2.5:3b
  .env harus berisi: GEMINI_API_KEY dan GROQ_API_KEY
        """,
    )
    parser.add_argument("--k",           type=int, default=5)
    parser.add_argument("--output",      type=str, default="hasil/hasil_evaluasi_rag.json")
    parser.add_argument("--csv",         type=str, default="hasil/hasil_retriever.csv")
    parser.add_argument("--csv-gen",     type=str, default="hasil/hasil_generator.csv")
    parser.add_argument("--skip-llm",    action="store_true")
    parser.add_argument("--llm",         type=str, default="all",
                        choices=["low", "medium", "high", "all"])
    parser.add_argument("--faithfulness", action="store_true")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv tidak terinstall")

    from rag import get_index_info, build_index
    if not Path("faiss_index.bin").exists():
        logger.info("Index belum ada, membangun index RAG dari knowledge_base/...")
        build_index()

    info = get_index_info()
    print("\n📚 RAG Index Info:")
    for k_info, v in info.items():
        if k_info != "supported_classes":
            print(f"   {k_info}: {v}")
    print(f"   supported_classes ({info['total_classes']}): {', '.join(info['supported_classes'])}")

    print(f"\n🔍 Evaluasi Retriever (k={args.k}, {len(GROUND_TRUTH_QA)} query)...")
    retriever_res = evaluate_retriever(GROUND_TRUTH_QA, k=args.k)
    print_retriever_table(retriever_res)
    save_retriever_csv(retriever_res, args.csv)

    generator_results: dict = {}
    if not args.skip_llm:
        llms_to_eval = []
        errors       = []

        run_low    = args.llm in ("low",    "all")
        run_medium = args.llm in ("medium", "all")
        run_high   = args.llm in ("high",   "all")

        if run_low:
            try:
                llms_to_eval.append(
                    _make_ollama_func(LLM_LOW_MODEL, f"Ollama — {LLM_LOW_MODEL} (LOW)")
                )
            except Exception as e:
                logger.warning(f"LOW (Qwen 3B) tidak bisa dimuat: {e}")
                errors.append(f"LOW: {e}")

        if run_medium:
            try:
                llms_to_eval.append(_make_gemini_func())
            except Exception as e:
                logger.warning(f"MEDIUM (Gemini) tidak bisa dimuat: {e}")
                errors.append(f"MEDIUM: {e}")

        if run_high:
            try:
                llms_to_eval.append(_make_groq_func())
            except Exception as e:
                logger.warning(f"HIGH (Llama 70B Groq) tidak bisa dimuat: {e}")
                errors.append(f"HIGH: {e}")

        if not llms_to_eval:
            logger.error("Tidak ada LLM yang bisa dijalankan. Cek .env & Ollama.")
            generator_results["errors"] = errors
        else:
            for llm_func, llm_name in llms_to_eval:
                print(f"\n🤖 Evaluasi Generator: {llm_name}...")
                try:
                    gen_res = evaluate_generator(GROUND_TRUTH_QA, llm_func, llm_name, k=args.k)
                    print_generator_table(gen_res)
                    if args.faithfulness:
                        gen_res = run_faithfulness_annotation(gen_res)
                        print_generator_table(gen_res)
                    generator_results[llm_name] = gen_res
                except Exception as e:
                    logger.error(f"Generator {llm_name} gagal: {e}")
                    generator_results[llm_name] = {"error": str(e)}

    save_results(retriever_res, generator_results, args.output)
    save_generator_csv(generator_results, args.csv_gen)

    print(f"\n✅ Selesai!")
    print(f"   JSON hasil    : {args.output}")
    print(f"   CSV retriever : {args.csv}")
    print(f"   CSV generator : {args.csv_gen}")

    llm_keys = [k for k in generator_results if "error" not in str(generator_results.get(k, {}))]
    if llm_keys:
        print(f"\n📊 Ringkasan Generator:")
        for lk in llm_keys:
            r = generator_results[lk]
            print(f"   [{lk}]")
            print(f"     ROUGE-1 F1 : {r.get('avg_rouge1_f', 'N/A')}")
            print(f"     ROUGE-2 F1 : {r.get('avg_rouge2_f', 'N/A')}")
            print(f"     ROUGE-L F1 : {r.get('avg_rougeL_f', 'N/A')}")
            print(f"     Cosine Sim : {r.get('avg_cosine_similarity', 'N/A')}")
            print(f"     Avg Waktu  : {r.get('avg_generation_time_s', 'N/A')} detik")


if __name__ == "__main__":
    main()