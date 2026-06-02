"""
isi_excel_untuk_validator.py
============================
Mengisi kolom jawaban Groq dan Gemini di Excel Ground Truth
dari hasil_evaluasi_rag.json secara otomatis.

Output: file Excel siap dikirim ke Kang Tangono untuk dinilai.

Cara pakai:
  python isi_excel_untuk_validator.py
"""

import json, re
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─── Konfigurasi ────────────────────────────────────────────────
JSON_PATH  = "hasil_evaluasi_rag.json"
OUTPUT     = "Penilaian_Faithfulness_untuk_Validator.xlsx"


# ─── Baca JSON ──────────────────────────────────────────────────
def load_json(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def extract_generators(d: dict) -> dict:
    """Kembalikan dict: {llm_name: per_query_list}"""
    result = {}
    if "generators" in d and isinstance(d["generators"], list):
        for g in d["generators"]:
            if "error" not in g:
                result[g["llm"]] = g.get("per_query", [])
    elif "generator" in d and isinstance(d["generator"], dict):
        for llm_name, g in d["generator"].items():
            if isinstance(g, dict) and "error" not in g:
                result[llm_name] = g.get("per_query", [])
    return result


def clean_answer(text: str, max_chars: int = 1500) -> str:
    """Bersihkan dan potong jawaban agar tidak terlalu panjang di Excel."""
    if not text:
        return "-"
    # Bersihkan whitespace berlebih
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...jawaban dipotong untuk keterbacaan...]"
    return text


# ─── Styling ────────────────────────────────────────────────────
def make_border():
    s = Side(style='thin')
    return Border(left=s, right=s, top=s, bottom=s)

def style(cell, bold=False, fill_hex=None, size=9, wrap=True,
          halign="left", valign="top", color="000000"):
    cell.font      = Font(bold=bold, size=size, name="Arial", color=color)
    cell.alignment = Alignment(horizontal=halign, vertical=valign,
                               wrap_text=wrap)
    cell.border    = make_border()
    if fill_hex:
        cell.fill  = PatternFill("solid", fgColor=fill_hex)


# ─── Build Excel ────────────────────────────────────────────────
def build_excel(d: dict, output_path: str):
    generators = extract_generators(d)
    if not generators:
        print("❌ Tidak ada data generator di JSON.")
        return

    # Buat ground truth dict dari retriever per_query sebagai acuan urutan
    # (atau ambil dari generator pertama)
    first_gen = next(iter(generators.values()))
    ids = [pq["id"] for pq in first_gen]

    wb = Workbook()

    # ── Sheet utama: Penilaian Faithfulness ─────────────────────
    ws = wb.active
    ws.title = "Penilaian Faithfulness"

    # Judul
    ws.merge_cells("A1:H1")
    ws["A1"] = "LEMBAR PENILAIAN FAITHFULNESS — Sistem Rekomendasi Penyakit Padi"
    style(ws["A1"], bold=True, fill_hex="1A7A4A", size=13,
          halign="center", color="FFFFFF")
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:H2")
    ws["A2"] = ("Validator: Bapak/Ibu dimohon membaca jawaban sistem dan membandingkan "
                "dengan referensi, lalu mengisi skor di kolom 'Skor Faithfulness' "
                "(0.0 = tidak akurat, 0.5 = sebagian benar, 1.0 = sangat akurat)")
    style(ws["A2"], fill_hex="E8F5E9", size=9, halign="center", color="2E7D32")
    ws.row_dimensions[2].height = 30

    # Header kolom
    llm_names = list(generators.keys())
    # Ambil nama pendek untuk header
    groq_name   = next((n for n in llm_names if "Groq"   in n or "groq"   in n), llm_names[0] if llm_names else "LLM 1")
    gemini_name = next((n for n in llm_names if "Gemini" in n or "gemini" in n), llm_names[1] if len(llm_names) > 1 else "LLM 2")

    headers = [
        ("No", 4),
        ("ID", 5),
        ("Penyakit", 22),
        ("Query / Pertanyaan", 35),
        ("Jawaban Referensi\n(Ground Truth dari Literatur)", 45),
        (f"Jawaban\n{groq_name[:30]}", 45),
        (f"Jawaban\n{gemini_name[:30]}", 45),
        ("Skor Faithfulness\nGroq (0.0–1.0)\n[ISI OLEH VALIDATOR]", 16),
        ("Skor Faithfulness\nGemini (0.0–1.0)\n[ISI OLEH VALIDATOR]", 16),
    ]

    for col, (header, width) in enumerate(headers, 1):
        ws.column_dimensions[get_column_letter(col)].width = width
        c = ws.cell(row=3, column=col, value=header)
        style(c, bold=True, fill_hex="1A7A4A", size=9,
              halign="center", valign="center", color="FFFFFF")
    ws.row_dimensions[3].height = 45

    # Data baris
    fills = ["FFFFFF", "F5F5F5"]
    for i, pq_ref in enumerate(first_gen, 1):
        row = i + 3
        qid = pq_ref["id"]

        # Ambil jawaban dari masing-masing LLM
        groq_ans   = "-"
        gemini_ans = "-"
        for llm_name, per_query in generators.items():
            pq_match = next((p for p in per_query if p["id"] == qid), None)
            if pq_match:
                ans = clean_answer(pq_match.get("generated", ""))
                if "Groq" in llm_name or "groq" in llm_name:
                    groq_ans   = ans
                elif "Gemini" in llm_name or "gemini" in llm_name:
                    gemini_ans = ans

        fill = fills[i % 2]
        row_data = [
            i,
            pq_ref["id"],
            pq_ref["disease"],
            pq_ref.get("query", ""),
            pq_ref.get("ground_truth", ""),
            groq_ans,
            gemini_ans,
            "",   # Skor Groq — diisi validator
            "",   # Skor Gemini — diisi validator
        ]

        for col, val in enumerate(row_data, 1):
            c = ws.cell(row=row, column=col, value=val)
            if col in (8, 9):   # Kolom skor — kuning mencolok
                style(c, fill_hex="FFF176", halign="center",
                      valign="center", size=11, bold=True)
            elif col in (5, 6, 7):  # Kolom teks panjang
                style(c, fill_hex=fill, size=8)
            else:
                style(c, fill_hex=fill, halign="center"
                      if col <= 3 else "left", size=9)
        ws.row_dimensions[row].height = 120

    ws.freeze_panes = "A4"

    # ── Sheet 2: Panduan Validator ───────────────────────────────
    ws2 = wb.create_sheet("Panduan untuk Validator")
    ws2.column_dimensions["A"].width = 20
    ws2.column_dimensions["B"].width = 70

    ws2.merge_cells("A1:B1")
    ws2["A1"] = "PANDUAN PENGISIAN LEMBAR PENILAIAN FAITHFULNESS"
    style(ws2["A1"], bold=True, fill_hex="1A7A4A", size=13,
          halign="center", color="FFFFFF")
    ws2.row_dimensions[1].height = 26

    panduan = [
        ("", ""),
        ("APA YANG DINILAI?", ""),
        ("Faithfulness",
         "Seberapa akurat jawaban sistem AI dibandingkan dengan jawaban referensi "
         "dari literatur pertanian (BB Padi, IRRI, Kementan)."),
        ("", ""),
        ("CARA MENGISI", ""),
        ("Langkah 1",
         "Buka sheet 'Penilaian Faithfulness'"),
        ("Langkah 2",
         "Baca kolom 'Jawaban Referensi' (Ground Truth dari literatur)"),
        ("Langkah 3",
         "Baca kolom 'Jawaban Groq' dan 'Jawaban Gemini' dari sistem AI"),
        ("Langkah 4",
         "Isi skor di kolom kuning 'Skor Faithfulness Groq' dan 'Skor Faithfulness Gemini'"),
        ("", ""),
        ("PANDUAN SKOR", ""),
        ("1.0",
         "SANGAT AKURAT — semua fakta, nama penyakit, nama obat/pestisida, "
         "dan langkah penanganan sesuai dengan referensi"),
        ("0.8",
         "AKURAT — sebagian besar benar, satu-dua detail kurang spesifik "
         "tapi tidak menyesatkan petani"),
        ("0.5",
         "SEBAGIAN BENAR — ada informasi penting yang terlewat atau "
         "berbeda dari referensi"),
        ("0.2",
         "KURANG AKURAT — sebagian besar informasi berbeda atau tidak lengkap"),
        ("0.0",
         "TIDAK AKURAT — jawaban tidak sesuai penyakit yang ditanya, "
         "atau informasi yang diberikan salah"),
        ("", ""),
        ("CONTOH PENILAIAN", ""),
        ("Skor 1.0",
         "Referensi: 'Semprot Kocide 77 WP dosis 2-3 g/L, kurangi nitrogen'\n"
         "Jawaban AI: 'Semprot bakterisida Kocide 77 WP dosis 2 g/L, kurangi pupuk urea'\n"
         "→ Semua fakta penting ada dan benar"),
        ("Skor 0.5",
         "Referensi: 'Semprot Kocide 77 WP dosis 2-3 g/L, kurangi nitrogen'\n"
         "Jawaban AI: 'Jaga kebersihan lahan dan hindari kelembaban tinggi'\n"
         "→ Benar tapi tidak menyebut pestisida spesifik"),
        ("Skor 0.0",
         "Referensi: 'Hawar Daun Bakteri — semprot bakterisida tembaga'\n"
         "Jawaban AI: memberikan penanganan untuk penyakit lain\n"
         "→ Tidak relevan dengan penyakit yang ditanya"),
        ("", ""),
        ("CATATAN", "Tidak perlu mengisi seluruh tabel dalam satu waktu. "
         "Bisa dicicil per beberapa baris sesuai waktu Bapak/Ibu."),
    ]

    for i, (k, v) in enumerate(panduan, 2):
        ck = ws2.cell(row=i, column=1, value=k)
        cv = ws2.cell(row=i, column=2, value=v)

        if k in ("APA YANG DINILAI?", "CARA MENGISI",
                 "PANDUAN SKOR", "CONTOH PENILAIAN", "CATATAN"):
            style(ck, bold=True, fill_hex="E8F5E9", color="1A7A4A", size=10)
            style(cv, bold=True, fill_hex="E8F5E9", color="1A7A4A", size=10)
        elif k.startswith("Langkah"):
            style(ck, bold=True, fill_hex="E3F2FD", size=9)
            style(cv, fill_hex="E3F2FD", size=9)
        elif k in ("1.0", "0.8", "0.5", "0.2", "0.0"):
            style(ck, bold=True, fill_hex="FFF9C4", halign="center", size=10)
            style(cv, fill_hex="FFF9C4", size=9)
        elif k.startswith("Skor "):
            style(ck, bold=True, fill_hex="F5F5F5", size=9)
            style(cv, fill_hex="F5F5F5", size=8)
        else:
            ws2.row_dimensions[i].height = 6
            continue

        ws2.row_dimensions[i].height = 50 if "\n" in str(v) else 20

    wb.save(output_path)
    print(f"\n✅ File Excel untuk validator berhasil dibuat: {output_path}")
    print(f"\nIsi file:")
    print(f"  Sheet 1 — 'Penilaian Faithfulness' : {len(first_gen)} baris data siap dinilai")
    print(f"  Sheet 2 — 'Panduan untuk Validator' : penjelasan cara mengisi")
    print(f"\nKirim file ini ke Kang Tangono.")
    print(f"Minta beliau mengisi kolom KUNING 'Skor Faithfulness' untuk Groq dan Gemini.")
    print(f"Setelah diisi, minta kembali file Excel-nya untuk dimasukkan ke laporan.")


def main():
    if not Path(JSON_PATH).exists():
        print(f"❌ File tidak ditemukan: {JSON_PATH}")
        return

    print(f"Membaca data dari: {JSON_PATH}")
    d = load_json(JSON_PATH)
    build_excel(d, OUTPUT)


if __name__ == "__main__":
    main()
