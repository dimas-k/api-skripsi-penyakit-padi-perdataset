# Paddy Disease Detection API

API deteksi penyakit daun padi menggunakan Deep Learning + LLaMA 3.3 (Groq).

---

## Struktur Folder

```
backend/
├── main.py               ← FastAPI utama (endpoints)
├── model.py              ← Load model PyTorch & prediksi
├── llm.py                ← Koneksi Groq LLaMA (rekomendasi & chatbot)
├── schemas.py            ← Struktur request & response
├── requirements.txt      ← Dependencies
├── .env                  ← API key & path model (jangan di-upload ke GitHub!)
└── efficientnet_b0_best.h5  ← File model (taruh di folder ini)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Isi file .env
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
MODEL_PATH=efficientnet_b0_best.h5
```
Dapatkan GROQ_API_KEY gratis di: https://console.groq.com

### 3. Taruh file model .h5 di folder backend/
```
backend/
└── efficientnet_b0_best.h5   ← taruh di sini
```

### 4. Jalankan server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Endpoints

| Method | Endpoint   | Fungsi                                      |
|--------|------------|---------------------------------------------|
| GET    | /health    | Cek status API & model aktif                |
| GET    | /classes   | Daftar kelas penyakit yang bisa dideteksi   |
| POST   | /detect    | Upload gambar → prediksi + rekomendasi LLM  |
| POST   | /chat      | Chatbot tanya jawab penyakit padi           |

---

## Dokumentasi Interaktif

Setelah server jalan, buka di browser:
```
http://localhost:8000/docs     ← Swagger UI
http://localhost:8000/redoc    ← ReDoc
```

---

## Contoh Request

### Deteksi Penyakit
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@daun_padi.jpg"
```

Response:
```json
{
  "predicted_class": "leaf_blast",
  "confidence": 94.32,
  "recommendation": "Blas Daun adalah penyakit yang disebabkan oleh..."
}
```

### Chatbot
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Bagaimana cara mencegah leaf blast?",
    "detected_disease": "leaf_blast"
  }'
```

Response:
```json
{
  "reply": "Untuk mencegah blas daun, beberapa langkah yang bisa dilakukan..."
}
```

---

## Ganti Model

Untuk ganti model, ubah MODEL_PATH di file .env:
```
# EfficientNet-B0
MODEL_PATH=efficientnet_b0_best.h5

# ResNet-50
MODEL_PATH=resnet50_best.h5

# Swin Transformer Tiny
MODEL_PATH=swin_tiny_best.h5
```
Tidak perlu ubah kode apapun — model akan terdeteksi otomatis.

---

## Kelas Penyakit yang Didukung

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
