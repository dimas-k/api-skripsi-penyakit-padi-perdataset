"""
evaluate_rag.py — Evaluasi RAG Pipeline Penyakit Padi
======================================================
Menghitung metrik evaluasi sesuai arahan dosen:

RETRIEVER METRICS (seperti di paper CottonBot):
  - MRR (Mean Reciprocal Rank)
  - Hit@k  (binary: ditemukan minimal 1 chunk relevan di top-k)
  - Precision@k

GENERATOR METRICS:
  - Faithfulness (manual scoring 0–1, opsional via --faithfulness)
  - Cosine Similarity (generated vs ground truth)
  - Average Generation Time (detik)

Cara pakai:
  python evaluate_rag.py --k 3 --llm both --output hasil_evaluasi_rag.json
  python evaluate_rag.py --k 5 --llm groq
  python evaluate_rag.py --k 5 --skip-llm                  (hanya retriever)
  python evaluate_rag.py --k 3 --llm both --faithfulness   (+ anotasi manual)

Changelog:
  v2 (Jun 2025): +Q16 bacterial_leaf_streak, +Q17-Q18 harvest_stage
                 Fix: from llm import → diganti import lokal aman
                 Fix: generator pakai args.k, bukan hardcode k=3
                 Fix: Hit@k menggantikan nama Recall@k yang misleading
                 Fix: Gemini fallback model list
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
GROQ_MODEL_NAME      = "llama-3.3-70b-versatile"

# Fallback list Gemini (urutan dicoba dari atas)
GEMINI_CANDIDATES = [
    ("gemini-2.5-flash", "v1beta"),
    ("gemini-2.5-flash", "v1"),
    ("gemini-2.0-flash", "v1"),
    ("gemini-2.0-flash", "v1beta"),
]

# System prompt lokal — tidak import dari llm.py (agar aman tanpa GEMINI_API_KEY)
_SYSTEM_PROMPT = (
    "Kamu adalah asisten pertanian yang membantu petani padi langsung di lapangan. "
    "Gunakan bahasa sederhana dan mudah dipahami. "
    "Jawab langsung ke solusi yang bisa dikerjakan hari ini. "
    "Gunakan informasi dari basis pengetahuan sebagai acuan utama. "
    "Selalu jawab dalam Bahasa Indonesia."
)


# ═════════════════════════════════════════════════════════════════
# GROUND TRUTH — 18 Skenario (14 kelas × minimal 1 query)
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
    # ── Q16 ── bacterial_leaf_streak ── BARU v2 ───────────────────
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
    # ── Q17 ── harvest_stage ── BARU v2 ───────────────────────────
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
    # ── Q18 ── harvest_stage (sensor) ── BARU v2 ─────────────────
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
# RETRIEVER EVALUATION — MRR, Hit@k, Precision@k
# ═════════════════════════════════════════════════════════════════
def evaluate_retriever(ground_truth_list: list[dict], k: int = 5) -> dict:
    """
    Evaluasi komponen retriever RAG.

    Rumus (sesuai paper CottonBot, Kandamali et al. 2025):
      MRR        = (1/N) * Σ (1/rank_i)  — rank chunk relevan pertama di top-k
      Hit@k      = proporsi query di mana minimal 1 chunk relevan ditemukan di top-k
                   (disebut Recall@k di beberapa paper closed-domain, binary 0/1 per query)
      Precision@k = rata-rata proporsi chunk relevan di top-k  (0..1)

    Catatan: Hit@k berbeda dari Recall@k klasik.
      - Hit@k (dipakai di sini): 1 jika ada ≥1 chunk relevan di top-k, 0 jika tidak.
      - Recall@k klasik: num_relevant_retrieved / total_relevant_in_corpus.
      Karena total_relevant tidak diketahui pasti, digunakan Hit@k.
    """
    from rag import retrieve

    mrr_scores       = []
    hit_scores       = []
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

        # ── Hit@k ─────────────────────────────────────────────────
        # Binary: 1 jika setidaknya 1 chunk relevan ditemukan di top-k
        hit = 1.0 if sum(relevance_flags) > 0 else 0.0
        hit_scores.append(hit)

        # ── Precision@k ───────────────────────────────────────────
        precision = sum(relevance_flags) / k
        precision_scores.append(precision)

        per_query_results.append({
            "id"             : qa["id"],
            "query"          : query[:70] + "..." if len(query) > 70 else query,
            "disease"        : qa["disease"],
            "rr"             : round(rr, 4),
            "hit_k"          : round(hit, 4),
            "precision_k"    : round(precision, 4),
            "relevance_flags": relevance_flags,
            "top1_score"     : round(retrieved[0]["score"], 4) if retrieved else 0,
        })

    return {
        "MRR"             : round(float(np.mean(mrr_scores)), 4),
        f"Hit@{k}"        : round(float(np.mean(hit_scores)), 4),
        f"Precision@{k}"  : round(float(np.mean(precision_scores)), 4),
        "per_query"       : per_query_results,
        "k"               : k,
        "n_queries"       : len(ground_truth_list),
    }


# ═════════════════════════════════════════════════════════════════
# LLM FUNCTIONS — tanpa import llm.py (aman meski GEMINI_API_KEY tidak diset)
# ═════════════════════════════════════════════════════════════════
def _make_groq_func():
    """
    Buat fungsi generator Groq tanpa mengimpor llm.py secara langsung.
    llm.py memiliki raise ValueError() di level modul yang akan crash
    jika GEMINI_API_KEY tidak diset di .env.
    """
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY belum diset di .env")
    client = Groq(api_key=api_key)

    def fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    return fn, f"Groq — {GROQ_MODEL_NAME}"


def _make_gemini_func():
    """
    Buat fungsi generator Gemini dengan fallback model list.
    Mencoba kandidat model dari GEMINI_CANDIDATES secara urut.
    """
    import requests

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY belum diset di .env")

    _base   = "https://generativelanguage.googleapis.com"
    _active = [None]   # mutable closure untuk menyimpan model yang berhasil

    def fn(prompt: str) -> str:
        # Urutkan: aktif duluan jika sudah pernah berhasil
        if _active[0]:
            ordered = [_active[0]] + [c for c in GEMINI_CANDIDATES if c != _active[0]]
        else:
            ordered = list(GEMINI_CANDIDATES)

        payload = {
            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
        }

        last_err = None
        for model, ver in ordered:
            url = f"{_base}/{ver}/models/{model}:generateContent?key={api_key}"
            for attempt in range(3):
                resp = requests.post(url, json=payload, timeout=60)
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
    return fn, f"Gemini — {active_name}"


# ═════════════════════════════════════════════════════════════════
# GENERATOR EVALUATION — Cosine Similarity + Waktu + Faithfulness
# ═════════════════════════════════════════════════════════════════
def evaluate_generator(
    ground_truth_list: list[dict],
    llm_func         : callable,
    llm_name         : str = "LLM",
    k                : int = 5,
) -> dict:
    """
    Evaluasi komponen generator (LLM) dalam RAG pipeline.

    Args:
        ground_truth_list: list skenario QA
        llm_func         : callable(prompt: str) -> str
        llm_name         : label LLM untuk output
        k                : jumlah chunk yang diambil saat build_rag_prompt (sinkron dengan args.k)

    Metrik:
    - Cosine Similarity  : kemiripan semantik antara jawaban LLM dan ground truth
    - Avg Generation Time: rata-rata waktu generate (detik)
    - Faithfulness       : skor manual annotator 0-1 (opsional, diisi terpisah)
    """
    from rag import build_rag_prompt
    from sentence_transformers import SentenceTransformer

    embed_model  = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cosim_scores = []
    time_scores  = []
    per_query_res = []

    logger.info(f"Mengevaluasi generator: {llm_name} ({len(ground_truth_list)} query, k={k})...")

    for qa in ground_truth_list:
        disease      = qa["disease"]
        ground_truth = qa["ground_truth"]

        # FIX: gunakan k dari args, bukan hardcode k=3
        prompt, retrieved = build_rag_prompt(disease_name=disease, k=k)

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
        if answer:
            vecs = embed_model.encode(
                [answer, ground_truth],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            cos_sim = cosine_similarity(vecs[0], vecs[1])
        else:
            cos_sim = 0.0
        cosim_scores.append(cos_sim)

        per_query_res.append({
            "id"          : qa["id"],
            "disease"     : disease,
            "query"       : qa["query"],
            "generated"   : answer[:300] + "..." if len(answer) > 300 else answer,
            "ground_truth": ground_truth[:300] + "..." if len(ground_truth) > 300 else ground_truth,
            "cosine_sim"  : round(cos_sim, 4),
            "time_s"      : t_gen,
            "faithfulness": None,  
        })

    return {
        "llm"                  : llm_name,
        "avg_cosine_similarity": round(float(np.mean(cosim_scores)), 4),
        "avg_generation_time_s": round(float(np.mean(time_scores)), 3),
        "faithfulness_note"    : "Diisi manual annotator (0=tidak akurat, 0.5=sebagian, 1=sangat akurat)",
        "per_query"            : per_query_res,
        "n_queries"            : len(ground_truth_list),
        "k_used"               : k,
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
    print(f"\n  {'ID':<5} {'Penyakit':<30} {'RR':>6} {'Hit':>5} {'Prec':>6} {'Top1':>6} Flags")
    print(f"  {'-'*70}")
    for pq in results["per_query"]:
        flags_str = str(pq["relevance_flags"])
        print(
            f"  {pq['id']:<5} {pq['disease']:<30} {pq['rr']:>6.3f} "
            f"{pq['hit_k']:>5.1f} {pq['precision_k']:>6.3f} {pq['top1_score']:>6.3f} {flags_str}"
        )


def print_generator_table(results: dict):
    print(f"\n{'='*65}")
    print(f"  HASIL EVALUASI GENERATOR: {results['llm']}")
    print(f"{'='*65}")
    print(f"  Avg Cosine Similarity  : {results['avg_cosine_similarity']:.4f}")
    print(f"  Avg Generation Time    : {results['avg_generation_time_s']:.3f} detik")
    print(f"  k chunks dipakai       : {results.get('k_used', 'N/A')}")
    if "avg_faithfulness" in results:
        print(f"  Avg Faithfulness       : {results['avg_faithfulness']:.4f}")
    print(f"{'='*65}")


# ═════════════════════════════════════════════════════════════════
# SAVE CSV & JSON
# ═════════════════════════════════════════════════════════════════
def save_retriever_csv(retriever_res: dict, output_path: str = "hasil_retriever.csv"):
    import csv
    k = retriever_res["k"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Penyakit", "Query",
            "RR", f"Hit@{k}", f"Precision@{k}",
            "Top1_Score", "Relevant_Flags",
        ])
        for pq in retriever_res["per_query"]:
            writer.writerow([
                pq["id"], pq["disease"], pq["query"],
                pq["rr"], pq["hit_k"], pq["precision_k"],
                pq["top1_score"],
                str(pq["relevance_flags"]),
            ])
        writer.writerow([])
        writer.writerow([
            "RATA-RATA", "", "",
            retriever_res["MRR"],
            retriever_res[f"Hit@{k}"],
            retriever_res[f"Precision@{k}"],
            "", "",
        ])
    logger.info(f"CSV retriever disimpan: {output_path}")


def save_results(
    retriever_res   : dict,
    generator_results: dict,   # dict of {llm_name: gen_res} — bisa 1 atau 2 LLM
    output_path     : str = "hasil_evaluasi_rag.json",
):
    output = {
        "metadata": {
            "timestamp"       : time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_queries"       : retriever_res["n_queries"],
            "k"               : retriever_res["k"],
            "embedding_model" : EMBEDDING_MODEL_NAME,
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
        description="Evaluasi RAG Pipeline Penyakit Padi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python evaluate_rag.py --k 3 --llm both --output hasil_evaluasi_rag.json
  python evaluate_rag.py --k 5 --llm groq
  python evaluate_rag.py --k 5 --skip-llm
  python evaluate_rag.py --k 3 --llm both --faithfulness
        """,
    )
    parser.add_argument("--k",           type=int, default=5,
                        help="Jumlah chunk top-k untuk retriever (default: 5)")
    parser.add_argument("--output",      type=str, default="hasil_evaluasi_rag.json",
                        help="Path output JSON (default: hasil_evaluasi_rag.json)")
    parser.add_argument("--csv",         type=str, default="hasil_retriever.csv",
                        help="Path output CSV retriever (default: hasil_retriever.csv)")
    parser.add_argument("--skip-llm",    action="store_true",
                        help="Hanya evaluasi retriever, skip generator LLM")
    parser.add_argument("--llm",         type=str, default="both",
                        choices=["groq", "gemini", "both"],
                        help="LLM yang dievaluasi (default: both)")
    parser.add_argument("--faithfulness",action="store_true",
                        help="Jalankan anotasi faithfulness manual setelah generator selesai")
    args = parser.parse_args()

    # ── Load .env ────────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv tidak terinstall; baca env dari OS langsung")

    # ── Pastikan RAG index sudah ada ────────────────────────────
    from rag import get_index_info, build_index
    if not Path("faiss_index.bin").exists():
        logger.info("Index belum ada, membangun index RAG dari knowledge_base/...")
        build_index()

    info = get_index_info()
    print("\n📚 RAG Index Info:")
    for k_info, v in info.items():
        if k_info != "supported_classes":   # terlalu panjang, ringkas
            print(f"   {k_info}: {v}")
    print(f"   supported_classes ({info['total_classes']}): {', '.join(info['supported_classes'])}")

    # ── Evaluasi Retriever ───────────────────────────────────────
    print(f"\n🔍 Evaluasi Retriever (k={args.k}, {len(GROUND_TRUTH_QA)} query)...")
    retriever_res = evaluate_retriever(GROUND_TRUTH_QA, k=args.k)
    print_retriever_table(retriever_res)
    save_retriever_csv(retriever_res, args.csv)

    # ── Evaluasi Generator (opsional) ────────────────────────────
    generator_results: dict = {}
    if not args.skip_llm:
        llms_to_eval = []
        errors       = []

        if args.llm in ("groq", "both"):
            try:
                llms_to_eval.append(_make_groq_func())
            except Exception as e:
                logger.warning(f"Groq tidak bisa dimuat: {e}")
                errors.append(f"Groq: {e}")

        if args.llm in ("gemini", "both"):
            try:
                llms_to_eval.append(_make_gemini_func())
            except Exception as e:
                logger.warning(f"Gemini tidak bisa dimuat: {e}")
                errors.append(f"Gemini: {e}")

        if not llms_to_eval:
            logger.error("Tidak ada LLM yang bisa dijalankan. Cek .env file.")
            generator_results["errors"] = errors
        else:
            for llm_func, llm_name in llms_to_eval:
                print(f"\n🤖 Evaluasi Generator: {llm_name}...")
                try:
                    gen_res = evaluate_generator(
                        GROUND_TRUTH_QA, llm_func, llm_name, k=args.k
                    )
                    print_generator_table(gen_res)

                    if args.faithfulness:
                        gen_res = run_faithfulness_annotation(gen_res)
                        print_generator_table(gen_res)

                    generator_results[llm_name] = gen_res
                except Exception as e:
                    logger.error(f"Generator {llm_name} gagal: {e}")
                    generator_results[llm_name] = {"error": str(e)}

    # ── Simpan semua hasil ───────────────────────────────────────
    save_results(retriever_res, generator_results, args.output)

    print(f"\n✅ Selesai!")
    print(f"   Hasil JSON : {args.output}")
    print(f"   Hasil CSV  : {args.csv}")
    print(f"   Salin angka dari CSV ke sheet 'Hasil Retriever RAG' di Excel Ground Truth.")

    if generator_results and "errors" not in generator_results:
        llm_keys = [k for k in generator_results if "error" not in str(generator_results[k])]
        if llm_keys:
            print(f"   Salin Cosine Sim + Waktu ke sheet 'Hasil Generator LLM+RAG'.")
        if args.faithfulness and llm_keys:
            print(f"   Faithfulness sudah diisi interaktif.")
        elif llm_keys:
            print(f"   Jalankan lagi dengan --faithfulness untuk mengisi skor Faithfulness secara manual.")


if __name__ == "__main__":
    main()
