"""
isi_excel_untuk_validator.py  (v2 — A4 Print-Ready)
=====================================================
Perubahan dari v1:
  - Layout dipadatkan untuk cetak A4 Landscape
  - Tambah kolom Ground Truth ke-2 (dari penyakit_padi.txt)
  - Jawaban 3 LLM DIGABUNG dalam 1 kolom (dibedakan label)
  - Jawaban disingkat max 350 karakter per LLM
  - Setup print: A4 landscape, fit 1 halaman lebar, margin tipis

3 Tier LLM:
  LOW    : Qwen2.5-3B        (Ollama lokal)
  MEDIUM : Gemini 2.5 Flash  (Google API)
  HIGH   : Llama3.3-70B      (Groq API)

Cara pakai:
  python isi_excel_untuk_validator.py
  python isi_excel_untuk_validator.py --input hasil_evaluasi_rag.json --kb knowledge_base/penyakit_padi.txt
"""

import json, re, argparse
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.page import PageMargins

# ─── Konfigurasi ────────────────────────────────────────────────
JSON_PATH  = "hasil_evaluasi_rag.json"
KB_PATH    = "knowledge_base/penyakit_padi.txt"
OUTPUT     = "Penilaian_Faithfulness_untuk_Validator.xlsx"

MAX_CHARS_GT    = 600   # max karakter ground truth per kolom
MAX_CHARS_ANS   = 350   # max karakter jawaban per LLM (digabung)

TIER_KEYWORDS = {
    "LOW"   : ["LOW",  "low",  "qwen", "Qwen", "3b", "3B"],
    "MEDIUM": ["MEDIUM", "medium", "Gemini", "gemini", "Flash"],
    "HIGH"  : ["HIGH", "high", "llama", "Llama", "70b", "70B",
               "versatile", "groq", "Groq"],
}

TIER_COLORS = {
    "LOW"   : "1565C0",
    "MEDIUM": "1A7A4A",
    "HIGH"  : "6A1B9A",
}

TIER_LIGHT = {
    "LOW"   : "BBDEFB",
    "MEDIUM": "C8E6C9",
    "HIGH"  : "E1BEE7",
}


# ═════════════════════════════════════════════════════════════════
# PARSE KNOWLEDGE BASE → Ground Truth ke-2
# ═════════════════════════════════════════════════════════════════
def parse_kb(kb_path: str) -> dict[str, str]:
    """
    Baca penyakit_padi.txt dan ekstrak GEJALA + PENANGANAN per kode penyakit.
    Kembalikan dict: {kode_penyakit: teks_gt2}

    Mendukung variasi header:
      - GEJALA:
      - PENANGANAN DAN PENGENDALIAN:
      - PENANGAHAN DAN PENGENDALIAN:  (typo di beberapa entri)
      - PENANGANAN PADA FASE PANEN:   (harvest_stage)
    Serta fallback untuk kelas non-penyakit:
      - CIRI-CIRI (healthy)
      - PEMELIHARAAN (healthy)
      - INDIKATOR KESIAPAN PANEN (harvest_stage)
    """
    text = Path(kb_path).read_text(encoding="utf-8")

    # Split per blok penyakit (dipisah baris ===...===)
    bloks = re.split(r"={3,}", text)

    result = {}
    kode      = None
    nama_id   = None

    for blok in bloks:
        # Cari baris "Kode: xxx"
        m = re.search(r"Kode:\s*(\w+)", blok)
        if m:
            kode = m.group(1).strip().lower()

        # Cari nama Indonesia (Nama: / Penyakit: / Nama Penyakit:)
        mn = re.search(r"(?:Nama(?:\s+Penyakit)?|Penyakit)\s*:\s*(.+)", blok)
        if mn:
            nama_id = mn.group(1).strip()

        if not kode:
            continue

        # ── Ekstrak GEJALA ─────────────────────────────────────
        gejala = ""

        # Pattern 1: Header GEJALA: standar
        mg = re.search(r"GEJALA\s*:\s*(.*?)(?=\n[A-Z]{3,}|\Z)", blok, re.S)
        if mg:
            gejala = mg.group(1).strip()

        # Fallback 1: CIRI-CIRI ... (untuk kelas healthy)
        if not gejala:
            mg = re.search(r"CIRI-CIRI[^:]*:\s*(.*?)(?=\n[A-Z]{3,}|\Z)", blok, re.S)
            if mg:
                gejala = "Ciri-ciri: " + mg.group(1).strip()

        # Fallback 2: INDIKATOR KESIAPAN PANEN (untuk kelas harvest_stage)
        if not gejala:
            mg = re.search(r"INDIKATOR[^:]*:\s*(.*?)(?=\n[A-Z]{3,}|\Z)", blok, re.S)
            if mg:
                gejala = "Indikator: " + mg.group(1).strip()

        if gejala:
            gejala = re.sub(r"\s+", " ", gejala)
            if len(gejala) > 280:
                gejala = gejala[:280].rsplit(" ", 1)[0] + "..."

        # ── Ekstrak PENANGANAN ──────────────────────────────────
        penanganan = ""

        # Pattern fleksibel: handle PENANGANAN / PENANGAHAN (typo) +
        # kata opsional setelahnya (DAN PENGENDALIAN / PADA FASE PANEN / dll.)
        mp = re.search(
            r"PENANG[A-Z]+(?:\s+[A-Z]+)*\s*:\s*(.*?)(?=\n[A-Z]{4,}|\Z)",
            blok, re.S
        )

        # Fallback: PEMELIHARAAN (untuk kelas healthy)
        if not mp:
            mp = re.search(
                r"PEMELIHARAAN[^:]*:\s*(.*?)(?=\n[A-Z]{4,}|\Z)",
                blok, re.S
            )

        if mp:
            raw = mp.group(1).strip()
            # Ambil hanya baris bernomor (1. 2. 3. dst)
            baris = re.findall(r"\d+\.\s+[^\n]+", raw)
            penanganan = " | ".join(b.strip() for b in baris[:3])
            if len(penanganan) > 320:
                penanganan = penanganan[:320].rsplit(" ", 1)[0] + "..."

        # ── Simpan hasil ────────────────────────────────────────
        if gejala or penanganan:
            # Header nama penyakit (Indonesia + kode Inggris)
            header = ""
            if nama_id:
                header = f"{nama_id} ({kode})"
            else:
                header = kode
            gt2 = f"[{header}]"
            if gejala:
                gt2 += f"\nGejala: {gejala}"
            if penanganan:
                gt2 += f"\nPenanganan: {penanganan}"
            result[kode] = gt2.strip()
            kode    = None   # reset setelah disimpan
            nama_id = None

    return result


# ═════════════════════════════════════════════════════════════════
# UTILS
# ═════════════════════════════════════════════════════════════════
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_generators(d: dict) -> list[dict]:
    raw = []
    if "generators" in d and isinstance(d["generators"], list):
        raw = [g for g in d["generators"] if "error" not in g]
    elif "generator" in d and isinstance(d["generator"], dict):
        raw = [g for g in d["generator"].values()
               if isinstance(g, dict) and "error" not in g]

    def tier_order(g):
        name = g.get("llm", "")
        for i, (tier, kws) in enumerate(TIER_KEYWORDS.items()):
            if any(kw in name for kw in kws):
                return i
        return 99

    return sorted(raw, key=tier_order)


def detect_tier(llm_name: str) -> str:
    for tier, kws in TIER_KEYWORDS.items():
        if any(kw in llm_name for kw in kws):
            return tier
    return "OTHER"


def short_name(llm_name: str, tier: str) -> str:
    """Nama pendek LLM untuk label di sel gabungan."""
    if tier == "LOW":
        return "Qwen2.5-3B"
    if tier == "MEDIUM":
        return "Gemini 2.5 Flash"
    if tier == "HIGH":
        return "Llama 3.3-70B"
    return llm_name[:20]


def shorten(text: str, max_chars: int = MAX_CHARS_ANS) -> str:
    if not text:
        return "-"
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    text = re.sub(r"\*\*|__", "", text)          # hapus markdown bold
    text = re.sub(r"\*|_", "", text)             # hapus markdown italic
    text = re.sub(r"#+\s+", "", text)            # hapus heading markdown
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def build_combined_answer(generators: list[dict], qid: str, tiers: list[str]) -> str:
    """
    Gabungkan jawaban semua LLM ke 1 string.
    Format:
      [LOW] Qwen2.5-3B
      ...jawaban singkat...

      [MEDIUM] Gemini 2.5 Flash
      ...jawaban singkat...

      [HIGH] Llama 3.3-70B
      ...jawaban singkat...
    """
    parts = []
    for gen, tier in zip(generators, tiers):
        pq_match = next((p for p in gen["per_query"] if p["id"] == qid), None)
        ans = shorten(pq_match.get("generated", "") if pq_match else "")
        sn  = short_name(gen["llm"], tier)
        parts.append(f"[{tier}] {sn}:\n{ans}")
    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════
# STYLING
# ═════════════════════════════════════════════════════════════════
def make_border(style_type="thin"):
    s = Side(style=style_type)
    return Border(left=s, right=s, top=s, bottom=s)


def cell_style(cell, bold=False, fill_hex=None, size=11, wrap=True,
               halign="left", valign="top", color="000000"):
    cell.font      = Font(bold=bold, size=size, name="Arial Narrow", color=color)
    cell.alignment = Alignment(horizontal=halign, vertical=valign,
                               wrap_text=wrap, shrink_to_fit=False)
    cell.border    = make_border()
    if fill_hex:
        cell.fill  = PatternFill("solid", fgColor=fill_hex)


# ═════════════════════════════════════════════════════════════════
# BUILD EXCEL
# ═════════════════════════════════════════════════════════════════
def build_excel(d: dict, kb_data: dict, output_path: str):
    generators = extract_generators(d)
    if not generators:
        print("❌ Tidak ada data generator di JSON.")
        return

    tiers     = [detect_tier(g["llm"]) for g in generators]
    first_gen = generators[0]["per_query"]

    wb = Workbook()

    # ════════════════════════════════════════════════════════════
    # SHEET 1: Penilaian Faithfulness
    # ════════════════════════════════════════════════════════════
    ws = wb.active
    ws.title = "Penilaian Faithfulness"

    # ── Setup halaman A4 Landscape ────────────────────────────
    ws.page_setup.orientation    = "landscape"
    ws.page_setup.paperSize      = 9          # A4
    ws.page_setup.fitToPage      = True
    ws.page_setup.fitToWidth     = 1          # muat 1 halaman lebar
    ws.page_setup.fitToHeight    = 0          # tinggi bebas
    ws.page_setup.scale          = 100
    ws.print_options.horizontalCentered = True
    ws.page_margins = PageMargins(
        left=0.4, right=0.4, top=0.5, bottom=0.5,
        header=0.2, footer=0.2
    )
    ws.oddHeader.center.text = (
        "LEMBAR PENILAIAN FAITHFULNESS — Sistem RAG Penyakit Padi"
    )
    ws.oddFooter.right.text = "Halaman &P dari &N"

    # ── Struktur kolom ───────────────────────────────────────
    # A=No | B=Penyakit | C=Query | D=GT-1 | E=GT-2 | F=Jawaban Gabungan
    # G=Skor LOW | H=Skor MEDIUM | I=Skor HIGH
    COL_NO    = 1
    COL_PYK   = 2
    COL_QRY   = 3
    COL_GT1   = 4
    COL_GT2   = 5
    COL_ANS   = 6
    COL_SKOR  = [7, 8, 9]   # satu per tier (LOW, MEDIUM, HIGH)
    TOTAL_COLS = 9

    col_widths = {
        COL_NO  : 4,
        COL_PYK : 18,
        COL_QRY : 26,
        COL_GT1 : 32,
        COL_GT2 : 32,
        COL_ANS : 52,
        7: 10, 8: 10, 9: 10,
    }
    for col, w in col_widths.items():
        ws.column_dimensions[get_column_letter(col)].width = w

    last_col = get_column_letter(TOTAL_COLS)

    # ── Baris 1: Judul ───────────────────────────────────────
    ws.merge_cells(f"A1:{last_col}1")
    ws["A1"] = "LEMBAR PENILAIAN FAITHFULNESS — Sistem Rekomendasi Penyakit Padi"
    cell_style(ws["A1"], bold=True, fill_hex="1B5E20", size=12,
               halign="center", valign="center", color="FFFFFF")
    ws.row_dimensions[1].height = 22

    # ── Baris 2: Instruksi ───────────────────────────────────
    ws.merge_cells(f"A2:{last_col}2")
    ws["A2"] = (
        "Validator: Baca Ground Truth (GT-1 + GT-2) lalu bandingkan dengan Jawaban LLM. "
        "Isi skor di kolom KUNING  "
        "0.0 = Tidak akurat   |   0.2 = Kurang akurat   |   "
        "0.5 = Sebagian benar   |   0.8 = Akurat   |   1.0 = Sangat akurat"
    )
    cell_style(ws["A2"], fill_hex="E8F5E9", size=8,
               halign="center", valign="center", color="1B5E20")
    ws.row_dimensions[2].height = 24

    # ── Baris 3: Header ──────────────────────────────────────
    header_dark = "263238"
    headers = [
        (COL_NO,  "No"),
        (COL_PYK, "Penyakit"),
        (COL_QRY, "Query / Pertanyaan"),
        (COL_GT1, "Ground Truth 1\n(dari Evaluasi RAG)"),
        (COL_GT2, "Ground Truth 2\n(dari Literatur Penyakit Padi)"),
        (COL_ANS, "Jawaban Sistem AI\n(LOW = Qwen2.5-3B  |  MEDIUM = Gemini  |  HIGH = Llama 3.3-70B)"),
    ]
    for col, label in headers:
        c = ws.cell(row=3, column=col, value=label)
        cell_style(c, bold=True, fill_hex=header_dark, size=8,
                   halign="center", valign="center", color="FFFFFF")

    # Header skor per tier
    for i, (tier, col) in enumerate(zip(["LOW", "MEDIUM", "HIGH"], COL_SKOR)):
        color_hex = TIER_COLORS.get(tier, "424242")
        sn = short_name("", tier)
        c  = ws.cell(row=3, column=col,
                     value=f"Skor\n[{tier}]\n{sn}\n(ISI VALIDATOR)")
        cell_style(c, bold=True, fill_hex=color_hex, size=7,
                   halign="center", valign="center", color="FFFFFF")

    ws.row_dimensions[3].height = 50
    ws.freeze_panes = "A4"

    # ── Baris data ───────────────────────────────────────────
    alt = ["FFFFFF", "F5F5F5"]

    for row_i, pq_ref in enumerate(first_gen, 1):
        row  = row_i + 3
        qid  = pq_ref["id"]
        dis  = pq_ref["disease"]
        fill = alt[row_i % 2]

        # GT-1 disingkat — tambah header nama penyakit
        gt1_raw  = pq_ref.get("ground_truth", "")
        gt1_body = gt1_raw if len(gt1_raw) <= MAX_CHARS_GT else \
                   gt1_raw[:MAX_CHARS_GT].rsplit(" ", 1)[0] + "…"
        # Ambil nama Indonesia dari kb_data untuk header GT-1
        kb_entry   = kb_data.get(dis.lower(), "")
        m_hdr      = re.match(r"\[(.+?)\]", kb_entry) if kb_entry else None
        gt1_header = m_hdr.group(1) if m_hdr else dis
        gt1 = f"[{gt1_header}]\n{gt1_body}"
        # GT-2 dari penyakit_padi.txt
        gt2 = kb_data.get(dis.lower(), "—")

        # Jawaban gabungan
        combined = build_combined_answer(generators, qid, tiers)

        data_cols = [
            (COL_NO,  row_i,   "center"),
            (COL_PYK, dis,     "left"),
            (COL_QRY, pq_ref.get("query", ""), "left"),
            (COL_GT1, gt1,     "left"),
            (COL_GT2, gt2,     "left"),
            (COL_ANS, combined,"left"),
        ]
        for col, val, ha in data_cols:
            c = ws.cell(row=row, column=col, value=val)
            cell_style(c, fill_hex=fill, size=11, halign=ha)

        # Kolom skor — kuning, diisi validator
        for i, (tier, col) in enumerate(zip(["LOW", "MEDIUM", "HIGH"], COL_SKOR)):
            c = ws.cell(row=row, column=col, value="")
            cell_style(c, fill_hex="FFF176", halign="center",
                       valign="center", size=11, bold=True)

        # Hitung tinggi baris dinamis berdasarkan kolom terpadat
        def est_lines(text, col_width_chars):
            if not text:
                return 1
            chars_per_line = max(1, int(col_width_chars * 1.35))
            lines = str(text).splitlines()
            return sum(max(1, -(-len(ln) // chars_per_line)) for ln in lines)

        max_lines = max(
            est_lines(gt1,      col_widths[COL_GT1]),
            est_lines(gt2,      col_widths[COL_GT2]),
            est_lines(combined, col_widths[COL_ANS]),
        )
        row_h = max(80, min(max_lines * 15 + 10, 500))
        ws.row_dimensions[row].height = row_h

    # Print area
    ws.print_area = f"A1:{last_col}{len(first_gen)+3}"

    # ════════════════════════════════════════════════════════════
    # SHEET 2: Panduan Validator
    # ════════════════════════════════════════════════════════════
    ws2 = wb.create_sheet("Panduan untuk Validator")
    ws2.page_setup.orientation = "portrait"
    ws2.page_setup.paperSize   = 9
    ws2.column_dimensions["A"].width = 22
    ws2.column_dimensions["B"].width = 78

    ws2.merge_cells("A1:B1")
    ws2["A1"] = "PANDUAN PENGISIAN LEMBAR PENILAIAN FAITHFULNESS"
    cell_style(ws2["A1"], bold=True, fill_hex="1B5E20", size=13,
               halign="center", color="FFFFFF")
    ws2.row_dimensions[1].height = 26

    tier_summary = "\n".join(
        f"  [{t}] → {short_name(g['llm'], t)}  ({g['llm']})"
        for g, t in zip(generators, tiers)
    )

    panduan = [
        ("", ""),
        ("APA YANG DINILAI?", ""),
        ("Faithfulness",
         "Seberapa akurat jawaban sistem AI dibandingkan dengan jawaban referensi "
         "dari literatur pertanian (BB Padi, IRRI, Kementan)."),
        ("", ""),
        ("3 TIER LLM\nyang Dievaluasi", tier_summary),
        ("", ""),
        ("2 SUMBER\nGROUND TRUTH", ""),
        ("Ground Truth 1",
         "Jawaban referensi yang dibuat peneliti berdasarkan literatur — "
         "digunakan sebagai acuan utama evaluasi otomatis (ROUGE, Cosine Similarity)."),
        ("Ground Truth 2",
         "Ringkasan GEJALA dan PENANGANAN dari file penyakit_padi.txt (BB Padi, IRRI, Kementan). "
         "Digunakan sebagai acuan tambahan agar validator dapat menilai dari dua sumber."),
        ("", ""),
        ("CARA MENGISI", ""),
        ("Langkah 1", "Buka sheet 'Penilaian Faithfulness'"),
        ("Langkah 2",
         "Baca kolom Ground Truth 1 DAN Ground Truth 2 sebagai referensi bersama"),
        ("Langkah 3",
         "Baca kolom 'Jawaban Sistem AI' — perhatikan label [LOW], [MEDIUM], [HIGH] "
         "untuk membedakan jawaban tiap model"),
        ("Langkah 4",
         "Isi skor di kolom KUNING untuk masing-masing tier (LOW / MEDIUM / HIGH)"),
        ("Langkah 5",
         "Skor boleh berbeda antar tier — nilai sesuai kualitas masing-masing jawaban"),
        ("", ""),
        ("PANDUAN SKOR", ""),
        ("1.0",
         "SANGAT AKURAT — semua fakta, nama penyakit, nama obat/pestisida, "
         "dan langkah penanganan sesuai dengan kedua referensi"),
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
        ("CATATAN",
         "• Tidak perlu mengisi seluruh tabel dalam satu waktu — bisa dicicil.\n"
         "• Jawaban AI sudah disingkat (max ±350 karakter) untuk kemudahan baca.\n"
         "• Kolom Jawaban berisi 3 model sekaligus, dipisah label [LOW]/[MEDIUM]/[HIGH].\n"
         "• Baris 'healthy' dan 'harvest_stage' BUKAN penyakit — nilai berdasarkan "
         "GT-1 dan GT-2 yang tersedia (penjelasan kondisi & penanganan optimal).\n"
         "• Setelah selesai, kirimkan kembali file Excel ini ke peneliti."),
    ]

    for i, (k, v) in enumerate(panduan, 2):
        ck = ws2.cell(row=i, column=1, value=k)
        cv = ws2.cell(row=i, column=2, value=v)

        if k in ("APA YANG DINILAI?", "CARA MENGISI", "PANDUAN SKOR",
                 "CONTOH PENILAIAN", "CATATAN", "2 SUMBER\nGROUND TRUTH",
                 "3 TIER LLM\nyang Dievaluasi"):
            cell_style(ck, bold=True, fill_hex="E8F5E9", color="1B5E20", size=10)
            cell_style(cv, bold=True, fill_hex="E8F5E9", color="1B5E20", size=10)
        elif k.startswith("Langkah"):
            cell_style(ck, bold=True, fill_hex="E3F2FD", size=9)
            cell_style(cv, fill_hex="E3F2FD", size=9)
        elif k in ("1.0", "0.8", "0.5", "0.2", "0.0"):
            cell_style(ck, bold=True, fill_hex="FFF9C4", halign="center", size=10)
            cell_style(cv, fill_hex="FFF9C4", size=9)
        elif k.startswith("Skor ") or k.startswith("Ground Truth") or k.startswith("Faithfulness"):
            cell_style(ck, bold=True, fill_hex="F5F5F5", size=9)
            cell_style(cv, fill_hex="F5F5F5", size=9)
        else:
            ws2.row_dimensions[i].height = 5
            continue

        ws2.row_dimensions[i].height = 55 if "\n" in str(v) else 20

    wb.save(output_path)

    print(f"\n✅ File Excel (v2 — A4 Print-Ready) berhasil dibuat: {output_path}")
    print(f"\nIsi file:")
    print(f"  Sheet 1 — 'Penilaian Faithfulness'")
    print(f"    {len(first_gen)} baris query × 3 LLM (jawaban digabung 1 kolom)")
    print(f"    Kolom: No | Penyakit | Query | GT-1 | GT-2 | Jawaban Gabungan | Skor×3")
    print(f"    GT-2 tersedia: {sum(1 for pq in first_gen if kb_data.get(pq['disease'].lower()))} dari {len(first_gen)} penyakit")
    for gen, tier in zip(generators, tiers):
        print(f"    [{tier}] {gen['llm']}")
    print(f"  Sheet 2 — 'Panduan untuk Validator'")
    print(f"\nCara cetak: File → Print → A4 Landscape → Fit Sheet on One Page (lebar)")
    print(f"Atau: Page Layout → Orientation: Landscape → Width: 1 page")


def main():
    parser = argparse.ArgumentParser(
        description="Buat Excel penilaian faithfulness (v2 — A4 Print-Ready)"
    )
    parser.add_argument("--input",  default=JSON_PATH, help="Path JSON hasil evaluasi")
    parser.add_argument("--kb",     default=KB_PATH,   help="Path knowledge base .txt")
    parser.add_argument("--output", default=OUTPUT,    help="Path output Excel")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ File tidak ditemukan: {args.input}")
        print(f"   Jalankan dulu: python evaluate_rag.py --k 3 --llm all")
        return

    print(f"Membaca JSON    : {args.input}")
    d = load_json(args.input)

    kb_data = {}
    if Path(args.kb).exists():
        print(f"Membaca KB      : {args.kb}")
        kb_data = parse_kb(args.kb)
        print(f"GT-2 ditemukan  : {len(kb_data)} penyakit dari knowledge base")
    else:
        print(f"⚠️  KB tidak ditemukan ({args.kb}), GT-2 akan kosong")

    build_excel(d, kb_data, args.output)


if __name__ == "__main__":
    main()