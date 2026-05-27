"""
rag.py — RAG Pipeline untuk Sistem Penyakit Padi
=================================================
Implementasi Retrieval-Augmented Generation (RAG) menggunakan:
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS
- Chunking: 1000 karakter dengan overlap 100 karakter
- Top-k retrieval: k=5 chunks
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
# KONFIGURASI
# ═════════════════════════════════════════════════════════════════
KNOWLEDGE_BASE_DIR  = Path(__file__).parent / "knowledge_base"
FAISS_INDEX_PATH    = Path(__file__).parent / "faiss_index.bin"
CHUNKS_PATH         = Path(__file__).parent / "chunks.pkl"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
CHUNK_SIZE          = 1000   # karakter per chunk
CHUNK_OVERLAP       = 100    # overlap antar chunk
TOP_K               = 5      # jumlah chunk yang diambil saat retrieval


# ═════════════════════════════════════════════════════════════════
# LAZY LOAD — model & index hanya dimuat saat pertama dipakai
# ═════════════════════════════════════════════════════════════════
_embedding_model = None
_faiss_index     = None
_chunks          = None      # list of str


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Memuat embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_index_and_chunks():
    """Load FAISS index + chunks dari disk, atau build jika belum ada."""
    global _faiss_index, _chunks
    if _faiss_index is None or _chunks is None:
        if FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists():
            _faiss_index, _chunks = _load_index()
        else:
            logger.info("Index belum ada, membangun ulang dari knowledge base...")
            _faiss_index, _chunks = build_index()
    return _faiss_index, _chunks


# ═════════════════════════════════════════════════════════════════
# STEP 1: CHUNKING DOKUMEN
# ═════════════════════════════════════════════════════════════════
def _load_documents() -> list[str]:
    """Muat semua file .txt dari folder knowledge_base."""
    docs = []
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Folder knowledge_base tidak ditemukan: {KNOWLEDGE_BASE_DIR}\n"
            f"Pastikan folder 'knowledge_base/' ada di direktori yang sama dengan rag.py"
        )
    for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        docs.append(text)
        logger.info(f"Dokumen dimuat: {file_path.name} ({len(text)} karakter)")
    if not docs:
        raise ValueError("Tidak ada file .txt di folder knowledge_base/")
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Memotong teks menjadi chunks dengan overlap.
    Strategi: potong pada batas newline terdekat untuk menjaga kohesi paragraf.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
            # Cari newline terdekat untuk memotong di batas yang lebih alami
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos + 1

        chunk = text[start:end].strip()
        if chunk:  # abaikan chunk kosong
            chunks.append(chunk)

        start = end - overlap  # mundur sebesar overlap untuk konteks

    return chunks


def _prepare_chunks(docs: list[str]) -> list[str]:
    """Gabungkan semua dokumen lalu potong menjadi chunks."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


# ═════════════════════════════════════════════════════════════════
# STEP 2: EMBEDDING & INDEXING
# ═════════════════════════════════════════════════════════════════
def _embed_texts(texts: list[str]) -> np.ndarray:
    """Ubah list of string menjadi matriks vektor (shape: N x D)."""
    model = _get_embedding_model()
    logger.info(f"Membuat embedding untuk {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalisasi → cosine similarity = dot product
    )
    return embeddings.astype(np.float32)


def build_index(save: bool = True):
    """
    Build FAISS index dari knowledge base.
    Dipanggil otomatis jika index belum ada, atau manual untuk rebuild.

    Returns:
        faiss_index: FAISS IndexFlatIP
        chunks: list[str]
    """
    import faiss

    docs   = _load_documents()
    chunks = _prepare_chunks(docs)
    embeds = _embed_texts(chunks)

    dim   = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product → cosine similarity (setelah normalisasi)
    index.add(embeds)

    if save:
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Index disimpan: {FAISS_INDEX_PATH} | {len(chunks)} chunks")

    return index, chunks


def _load_index():
    """Load FAISS index dan chunks dari file cache."""
    import faiss
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Index dimuat dari cache: {len(chunks)} chunks")
    return index, chunks


def rebuild_index():
    """Paksa rebuild index (gunakan setelah menambah dokumen baru ke knowledge_base/)."""
    global _faiss_index, _chunks
    logger.info("Rebuild index dimulai...")
    _faiss_index, _chunks = build_index(save=True)
    logger.info("Rebuild selesai.")
    return _faiss_index, _chunks


# ═════════════════════════════════════════════════════════════════
# STEP 3: RETRIEVAL
# ═════════════════════════════════════════════════════════════════
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Cari top-k chunks paling relevan untuk query.

    Args:
        query: pertanyaan atau nama penyakit
        k: jumlah chunk yang dikembalikan

    Returns:
        list of dict: [{"rank": int, "score": float, "text": str}, ...]
    """
    model            = _get_embedding_model()
    index, chunks    = _get_index_and_chunks()

    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_vec, k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx >= 0:  # FAISS mengembalikan -1 jika tidak cukup hasil
            results.append({
                "rank" : rank,
                "score": float(score),
                "text" : chunks[idx],
            })

    return results


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Format chunks hasil retrieval menjadi string konteks untuk LLM.
    """
    if not retrieved_chunks:
        return "Tidak ada konteks relevan ditemukan."

    parts = []
    for item in retrieved_chunks:
        parts.append(f"[Chunk {item['rank']} | Relevansi: {item['score']:.3f}]\n{item['text']}")

    return "\n\n---\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════
# STEP 4: AUGMENTED PROMPT BUILDER
# ═════════════════════════════════════════════════════════════════
def build_rag_prompt(
    disease_name: str,
    sensor_data : Optional[dict] = None,
    k           : int = TOP_K,
) -> tuple[str, list[dict]]:
    """
    Bangun prompt yang sudah diperkaya dengan konteks dari knowledge base (RAG).

    Args:
        disease_name: nama penyakit yang terdeteksi (kode atau nama Indonesia)
        sensor_data : dict data sensor IoT
        k           : jumlah chunks yang diambil

    Returns:
        (prompt_string, retrieved_chunks)
    """
    # Query menggabungkan nama penyakit dengan konteks sensor
    query = f"penanganan penyakit {disease_name} pada tanaman padi"
    if sensor_data:
        suhu = sensor_data.get("suhu_udara", "")
        kelembaban = sensor_data.get("kelembaban_udara", "")
        if suhu:
            query += f" suhu {suhu} derajat"
        if kelembaban:
            query += f" kelembaban {kelembaban} persen"

    retrieved_chunks = retrieve(query, k=k)
    context          = format_context(retrieved_chunks)

    sensor_section = ""
    if sensor_data:
        sensor_section = f"""
Data Sensor IoT Lapangan Saat Ini:
- Suhu Udara       : {sensor_data.get('suhu_udara', 'N/A')} °C
- Kelembaban Udara : {sensor_data.get('kelembaban_udara', 'N/A')} %
- Suhu Tanah       : {sensor_data.get('suhu_tanah', 'N/A')} °C
- Kelembaban Tanah : {sensor_data.get('kelembaban_tanah', 'N/A')} %
- pH Tanah         : {sensor_data.get('ph_tanah', 'N/A')}
- Nitrogen (N)     : {sensor_data.get('nitrogen', 'N/A')} mg/kg
- Fosfor (P)       : {sensor_data.get('fosfor', 'N/A')} mg/kg
- Kalium (K)       : {sensor_data.get('kalium', 'N/A')} mg/kg
- Intensitas Cahaya: {sensor_data.get('intensitas_cahaya', 'N/A')} lux
- Curah Hujan      : {sensor_data.get('curah_hujan', 'N/A')} mm/hari
"""

    prompt = f"""Berikut adalah informasi dari basis pengetahuan penyakit padi yang relevan:

{context}

---

Kamera AI mendeteksi penyakit: **{disease_name}**
{sensor_section}
Berdasarkan informasi di atas, bantu petani dengan:
1. Penjelasan singkat penyakit ini (gejala dan penyebab)
2. Langkah penanganan yang harus dilakukan SEKARANG
3. Cara pencegahan agar tidak terulang
4. Apakah kondisi sensor saat ini memperparah atau mendukung penyebaran? (jika data sensor tersedia)

Jawab dengan bahasa sederhana yang mudah dipahami petani dan langsung bisa diterapkan.
"""
    return prompt, retrieved_chunks


# ═════════════════════════════════════════════════════════════════
# HELPER: Status index
# ═════════════════════════════════════════════════════════════════
def get_index_info() -> dict:
    """Kembalikan info tentang index yang sedang aktif."""
    index, chunks = _get_index_and_chunks()
    model         = _get_embedding_model()
    return {
        "total_chunks"   : len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size"     : CHUNK_SIZE,
        "chunk_overlap"  : CHUNK_OVERLAP,
        "top_k"          : TOP_K,
        "index_type"     : "FAISS IndexFlatIP (cosine similarity)",
        "index_cached"   : FAISS_INDEX_PATH.exists(),
    }
