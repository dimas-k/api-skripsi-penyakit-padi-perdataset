# Paddy Disease Detection API

API deteksi penyakit daun padi menggunakan Deep Learning (Swin Transformer) + LLM (Groq LLaMA 3.3 & Gemini) dengan RAG.

- **Deployment** memakai **satu model gabungan** (Swin-B, 14 kelas) di endpoint `/predict`.
- **Riset/pembanding** memakai 20 model per-dataset (5 arsitektur × 4 dataset) di endpoint `/compare` & `/predict/{model_key}`.

---

## Struktur Folder

```
backend/
├── main.py                     ← FastAPI utama (endpoints)
├── model.py                    ← Load model PyTorch & prediksi
├── llm.py                      ← Koneksi Groq & Gemini (rekomendasi & chatbot)
├── rag.py                      ← Retrieval Augmented Generation
├── database.py                 ← Koneksi Supabase
├── schemas.py                  ← Struktur request & response
├── requirements.txt            ← Dependencies
├── .env                        ← API key & path model (JANGAN di-upload ke GitHub!)
│
├── model_gabungan/             ← Model utama untuk deployment
│   └── swin_base_best.h5        (Swin-B, 14 kelas)
│
├── models_perdataset/          ← 20 model per-dataset (khusus riset)
│   ├── swin_base_Citra_Daun_Padi_best.h5
│   ├── swin_base_JENIS_PENYAKIT_PADI_best.h5
│   ├── swin_base_paddy-dataset-v3-augmentasi_best.h5
│   ├── swin_base_Paddy-disease-classification_best.h5
│   ├── efficientnet_b0_...  inception_v3_...  resnet50_...  vit_...
│   └── (5 arsitektur × 4 dataset = 20 file .h5)
│
├── benchmark/                  ← Script benchmark model (riset)
│   ├── benchmark_model.py
│   └── benchmark_realcase.py
├── evaluasi_rag/               ← Script evaluasi RAG & faithfulness (riset)
│   ├── evaluate_rag.py / evaluate_rag_petani.py
│   ├── annotate_faithfulness.py / hitung_metrik_setelah_validasi.py
│   ├── isi_excel_*.py
│   └── check_gemini.py
└── hasil/                      ← Semua output: hasil_*.csv/xlsx/json, Penilaian_*.xlsx
```

> **Catatan:** script di `benchmark/` & `evaluasi_rag/` dijalankan **dari root proyek**
> (mis. `python evaluasi_rag/evaluate_rag.py`). Input/output default-nya sudah
> mengarah ke folder `hasil/`, dan knowledge base tetap dibaca dari `knowledge_base/`.

> **Format nama file per-dataset:** `{arsitektur}_{dataset}_best.h5`
> dengan `dataset` ∈ `Citra_Daun_Padi`, `JENIS_PENYAKIT_PADI`, `paddy-dataset-v3-augmentasi`, `Paddy-disease-classification`.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Buat file .env
Salin dari `env.example`, lalu isi:
```
SUPABASE_URL=...
SUPABASE_KEY=...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...

# Model gabungan (deployment)
MODEL_GABUNGAN_PATH=model_gabungan/swin_base_best.h5

# (Opsional) folder model per-dataset untuk endpoint riset
PERDATASET_DIR=models_perdataset
```

### 3. Taruh file model
- **Wajib (deployment):** `model_gabungan/swin_base_best.h5`
- **Opsional (riset):** 20 file di `models_perdataset/`

### 4. Jalankan server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Endpoints

| Method | Endpoint               | Fungsi                                                        |
|--------|------------------------|--------------------------------------------------------------|
| GET    | /health                | Cek status API & model aktif                                 |
| GET    | /classes               | Daftar kelas penyakit yang bisa dideteksi                    |
| GET    | /models                | Daftar 20 model per-dataset yang ter-load (riset)            |
| POST   | /predict               | **Deteksi pakai MODEL GABUNGAN** + rekomendasi LLM (deploy)  |
| POST   | /predict/{model_key}   | Deteksi pakai 1 model per-dataset tertentu (riset)           |
| POST   | /compare               | Bandingkan seluruh 20 model per-dataset (riset)              |
| POST   | /compare/by-arch       | Bandingkan model dikelompokkan per arsitektur (riset)        |
| POST   | /chat                  | Chatbot tanya jawab penyakit padi                            |
| GET    | /sensor                | Data sensor lingkungan (simulasi)                            |

---

## Dokumentasi Interaktif

Setelah server jalan, buka di browser:
```
http://localhost:8000/docs     ← Swagger UI
http://localhost:8000/redoc    ← ReDoc
```

---

## Contoh Request

### Deteksi Penyakit (deployment — model gabungan)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@daun_padi.jpg" \
  -H "x-user-id: <device-uuid>"
```

### Deteksi dengan 1 model per-dataset (riset)
```bash
curl -X POST http://localhost:8000/predict/swin_base__paddy_dataset_v3 \
  -F "file=@daun_padi.jpg"
```

### Bandingkan semua model (riset)
```bash
curl -X POST http://localhost:8000/compare \
  -F "file=@daun_padi.jpg"
```

---

## Kelas Penyakit yang Didukung (14 kelas — model gabungan)

| Nama Kelas                | Nama Indonesia              |
|---------------------------|-----------------------------|
| bacterial_leaf_blight     | Hawar Daun Bakteri          |
| bacterial_leaf_streak     | Hawar Daun Bergaris Bakteri |
| bacterial_panicle_blight  | Hawar Malai Bakteri         |
| brown_spot                | Bercak Coklat               |
| dead_heart                | Batang Mati                 |
| downy_mildew              | Embun Bulu                  |
| healthy                   | Sehat                       |
| hispa                     | Hispa Padi                  |
| leaf_blast                | Blas Daun                   |
| leaf_smut                 | Gosong Palsu Daun           |
| neck_blast                | Blas Leher Malai            |
| sheath_blight             | Busuk Pelepah               |
| tungro                    | Tungro                      |
| harvest_stage             | Fase Panen                  |
