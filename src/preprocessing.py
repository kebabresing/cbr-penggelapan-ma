# =====================================================
# preprocessing.py - Stage 1: Case Base Construction
# Sistem CBR Putusan Pengadilan - Pidana Penggelapan
# =====================================================

import os
import re
import logging
import pdfplumber
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_pdf(filepath):
    """
    Membaca isi teks dari satu file PDF menggunakan pdfplumber.
    Return: str berisi teks gabungan semua halaman
    """
    text_pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        return "\n".join(text_pages)
    except Exception as e:
        logger.error(f"Gagal membaca {filepath}: {e}")
        return ""


def clean_text(raw_text):
    """
    Membersihkan teks mentah hasil ekstraksi PDF.
    Langkah: lowercase -> hapus nomor halaman -> hapus header/footer
             -> hapus karakter khusus -> normalisasi spasi
    """
    if not raw_text:
        return ""
    text = raw_text.lower()

    # Hapus nomor halaman
    text = re.sub(r'halaman\s+\d+\s+(dari|of)\s+\d+', '', text)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'page\s+\d+', '', text)

    # Hapus header/footer berulang khas putusan MA
    patterns_to_remove = [
        r'mahkamah agung republik indonesia',
        r'direktori putusan',
        r'putusan\.mahkamahagung\.go\.id',
        r'kepaniteraan mahkamah agung[^\n]*\n',
        r'disclaimer[^\n]*\n',
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Hapus karakter khusus, pertahankan alfanumerik + tanda baca dasar
    text = re.sub(r'[^\w\s.,;:()\-/]', ' ', text)

    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_all_pdfs(raw_folder, output_path):
    """
    Memproses semua PDF dalam folder dan simpan ke CSV.
    Kolom output: case_id, filename, cleaned_text
    """
    raw_folder = Path(raw_folder)
    pdf_files = sorted(raw_folder.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"Tidak ada PDF di: {raw_folder}")
        return pd.DataFrame()

    logger.info(f"Ditemukan {len(pdf_files)} file PDF")
    records = []

    for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Memproses PDF")):
        case_id = f"CASE_{idx+1:03d}"
        raw_text = read_pdf(str(pdf_path))
        cleaned = clean_text(raw_text)

        if len(cleaned) < 100:
            logger.warning(f"Teks terlalu pendek, skip: {pdf_path.name}")
            continue

        records.append({
            "case_id": case_id,
            "filename": pdf_path.name,
            "cleaned_text": cleaned
        })

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Tersimpan {len(df)} kasus -> {output_path}")
    return df


if __name__ == "__main__":
    # Proses semua PDF asli dari folder data/raw/
    df = process_all_pdfs("data/raw", "data/processed/cleaned_cases.csv")
    if len(df) == 0:
        print("Tidak ada PDF yang berhasil diproses. Pastikan folder data/raw/ berisi file PDF.")
    else:
        print(df.head())
        print(f"Total kasus: {len(df)}")
