# =====================================================
# representation.py - Stage 2: Case Representation
# Sistem CBR Putusan Pengadilan - Pidana Penggelapan
# =====================================================

import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# FUNGSI 1: Ekstraksi metadata dari teks putusan
# -------------------------------------------------------
def extract_metadata(text):
    """
    Mengekstrak metadata penting dari teks putusan pengadilan.
    Menggunakan regex untuk menemukan pola umum di putusan MA.

    Return dict dengan kunci:
        nomor_perkara, tanggal_putusan, pasal, nama_terdakwa, nama_hakim
    """
    metadata = {
        "nomor_perkara": "",
        "tanggal_putusan": "",
        "pasal": "",
        "nama_terdakwa": "",
        "nama_hakim": ""
    }

    if not text:
        return metadata

    # --- Nomor Perkara ---
    # Pola: "No. 123/Pid.B/2023/PN.Jkt.Sel"
    perkara_pattern = r'(no\.?\s*\d+[\w./\-]+(?:pid|pdt)[\w./\-]+)'
    match = re.search(perkara_pattern, text, re.IGNORECASE)
    if match:
        metadata["nomor_perkara"] = match.group(1).strip()

    # --- Tanggal Putusan ---
    # Pola: "21 Januari 2023" atau "21-01-2023"
    bulan_map = {
        'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
        'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
        'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
    }
    tanggal_pattern = r'(\d{1,2})\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+(\d{4})'
    match = re.search(tanggal_pattern, text, re.IGNORECASE)
    if match:
        hari, bulan_str, tahun = match.groups()
        bulan = bulan_map.get(bulan_str.lower(), '00')
        metadata["tanggal_putusan"] = f"{tahun}-{bulan}-{int(hari):02d}"

    # --- Pasal yang Dilanggar ---
    # Pola: "pasal 372", "pasal 374 kuhp", "pasal 372 jo 55"
    pasal_pattern = r'pasal\s+(\d+(?:\s*(?:jo\.?\s*)?\d+)*)\s*(?:kuhp|kitab)?'
    matches = re.findall(pasal_pattern, text, re.IGNORECASE)
    if matches:
        # Ambil pasal yang paling sering muncul
        from collections import Counter
        most_common = Counter(matches).most_common(1)[0][0]
        metadata["pasal"] = f"Pasal {most_common.strip()} KUHP"

    # --- Nama Terdakwa ---
    # Pola: "terdakwa [NAMA]" atau "nama: [NAMA]"
    terdakwa_pattern = r'(?:nama\s+terdakwa|terdakwa\s*:?\s*)([A-Z][a-zA-Z\s]{3,30}?)(?:\s*(?:bin|binti|alias|,|\.))'
    match = re.search(terdakwa_pattern, text, re.IGNORECASE)
    if match:
        metadata["nama_terdakwa"] = match.group(1).strip().title()

    # --- Nama Hakim ---
    # Pola: "ketua majelis [NAMA]" atau "hakim ketua [NAMA]"
    hakim_pattern = r'(?:hakim ketua|ketua majelis)\s*:?\s*([A-Z][a-zA-Z\s]{3,30}?)(?:\s*,|\s*sh|\s*s\.h)'
    match = re.search(hakim_pattern, text, re.IGNORECASE)
    if match:
        metadata["nama_hakim"] = match.group(1).strip().title()

    return metadata


# -------------------------------------------------------
# FUNGSI 2: Ekstraksi fitur statistik teks
# -------------------------------------------------------
def extract_text_features(text):
    """
    Menghitung fitur statistik dari teks dokumen.

    Return dict: jumlah_kata, jumlah_kalimat, panjang_dokumen
    """
    if not text:
        return {"jumlah_kata": 0, "jumlah_kalimat": 0, "panjang_dokumen": 0}

    # Hitung kata (split by spasi)
    words = text.split()
    jumlah_kata = len(words)

    # Hitung kalimat (split by titik, tanda tanya, tanda seru)
    kalimat = re.split(r'[.!?]+', text)
    kalimat = [k.strip() for k in kalimat if len(k.strip()) > 10]
    jumlah_kalimat = len(kalimat)

    # Panjang dokumen (karakter)
    panjang_dokumen = len(text)

    return {
        "jumlah_kata": jumlah_kata,
        "jumlah_kalimat": jumlah_kalimat,
        "panjang_dokumen": panjang_dokumen
    }


# -------------------------------------------------------
# FUNGSI 3: TextRank sederhana untuk ringkasan
# -------------------------------------------------------
def simple_textrank_summary(text, n_sentences=3):
    """
    Membuat ringkasan teks menggunakan pendekatan TextRank sederhana.
    Algoritma:
    1. Pecah teks menjadi kalimat
    2. Hitung skor tiap kalimat berdasarkan frekuensi kata penting
    3. Ambil top-N kalimat sebagai ringkasan

    Parameter:
        text (str): Teks yang akan diringkas
        n_sentences (int): Jumlah kalimat ringkasan

    Return:
        str: Ringkasan teks
    """
    if not text or len(text) < 100:
        return text[:200] if text else ""

    # Kata-kata penting dalam putusan penggelapan
    kata_penting = {
        'penggelapan', 'terbukti', 'terdakwa', 'pidana', 'penjara',
        'pasal', 'kuhp', 'menggelapkan', 'uang', 'jabatan', 'kerugian',
        'putusan', 'hakim', 'dakwaan', 'hukuman', 'bersalah', 'dipidana'
    }

    # Pecah menjadi kalimat
    kalimat_list = re.split(r'(?<=[.!?])\s+', text)
    kalimat_list = [k.strip() for k in kalimat_list if len(k.strip()) > 20]

    if len(kalimat_list) <= n_sentences:
        return text[:500]

    # Hitung skor tiap kalimat
    skor = []
    for kalimat in kalimat_list:
        kata_kalimat = set(kalimat.lower().split())
        # Skor = jumlah kata penting yang ada di kalimat
        score = len(kata_kalimat.intersection(kata_penting))
        skor.append(score)

    # Ambil indeks kalimat dengan skor tertinggi (tetap urut posisi)
    top_indices = sorted(
        sorted(range(len(skor)), key=lambda i: skor[i], reverse=True)[:n_sentences]
    )

    ringkasan = '. '.join([kalimat_list[i] for i in top_indices])
    return ringkasan


# -------------------------------------------------------
# FUNGSI 4: Ekstraksi amar putusan
# -------------------------------------------------------
def extract_amar_putusan(text):
    """
    Mengekstrak bagian amar putusan dari teks.
    Mencari kata kunci: 'mengadili', 'memutuskan', 'amar putusan'
    dan mengambil teks setelahnya.

    Return:
        str: Amar putusan atau ringkasan teks akhir
    """
    if not text:
        return ""

    # Kata kunci yang menandai amar putusan
    keywords = ['mengadili', 'memutuskan', 'amar putusan', 'menyatakan terdakwa']

    for kw in keywords:
        idx = text.lower().find(kw)
        if idx != -1:
            # Ambil 300 karakter setelah kata kunci
            amar = text[idx:idx+400].strip()
            return amar

    # Jika tidak ditemukan, ambil 300 karakter terakhir
    return text[-300:].strip()


# -------------------------------------------------------
# FUNGSI 5: Klasifikasi kategori hukuman
# -------------------------------------------------------
def classify_hukuman(text):
    """
    Mengklasifikasikan berat hukuman berdasarkan teks putusan.
    Kategori:
        ringan : hukuman <= 2 tahun (<= 24 bulan)
        berat  : hukuman > 2 tahun (> 24 bulan)

    Dasar hukum:
        Pasal 372 KUHP (penggelapan) : maks 4 tahun
        Pasal 374 KUHP (penggelapan jabatan) : maks 5 tahun
        Threshold 2 tahun = titik tengah (median) dari rentang hukuman

    Pendekatan:
    - Mencari durasi hukuman HANYA dalam konteks kalimat yang mengandung
      kata kunci pidana (pidana penjara, menjatuhkan, dll)
    - Menghindari false positive dari referensi tanggal/pasal
      (misal: "48 tahun 2009" bukan hukuman)
    - Mengambil durasi MAKSIMUM dari semua match (putusan tertinggi)

    Return:
        str: Kategori hukuman (ringan / berat / tidak_diketahui)
    """
    if not text:
        return "tidak_diketahui"

    # Kata kunci yang menandakan konteks hukuman/pidana
    pidana_keywords = [
        'pidana penjara', 'pidana kurungan', 'penjara',
        'menjatuhkan', 'dijatuhi', 'menghukum',
        'dipidana', 'hukuman penjara'
    ]

    # Pola durasi hukuman (dengan toleransi OCR error: max 3 char antara angka dan satuan)
    # Match: "2 tahun", "2 (dua) tahun", "1 tahun 6 bulan", dll
    pola_tahun = r'(\d+)\s*(?:\([^)]*\))?\s*(?:tahun|thn)'
    pola_bulan = r'(\d+)\s*(?:\([^)]*\))?\s*(?:bulan|bln)'

    max_bulan = 0

    # --- STRATEGI 1: Cari per-kalimat (split by titik) ---
    # Setiap kalimat yang mengandung kata kunci pidana + durasi = kandidat hukuman
    kalimat_list = re.split(r'(?<=[.!?;])\s+', text.replace('\n', ' '))
    for kalimat in kalimat_list:
        kalimat_lower = kalimat.lower()
        has_keyword = any(kw in kalimat_lower for kw in pidana_keywords)
        if not has_keyword:
            continue

        # Hitung total durasi dalam kalimat ini
        total = 0
        m_tahun = re.search(pola_tahun, kalimat, re.IGNORECASE)
        if m_tahun:
            total += int(m_tahun.group(1)) * 12
        m_bulan = re.search(pola_bulan, kalimat, re.IGNORECASE)
        if m_bulan:
            total += int(m_bulan.group(1))

        if total > max_bulan:
            max_bulan = total

    # --- STRATEGI 2 (Fallback): Cari pola "pidana penjara selama X" ---
    # Menangkap kasus dimana split kalimat tidak sempurna karena OCR
    if max_bulan == 0:
        pola_selama = (
            r'pidana\s+(?:penjara|kurungan)\s+selama\s+(\d+)'
            r'|'
            r'selama\s+(\d+)\s*(?:\([^)]*\))?\s*(?:tahun|bulan|thn|bln)'
            r'\s*(?:.*?pidana\s+(?:penjara|kurungan))?'
        )
        for match in re.finditer(pola_selama, text, re.IGNORECASE):
            num = match.group(1) or match.group(2)
            if num:
                # Tentukan satuan dari konteks sekitar angka
                surrounding = text[match.start():min(match.end() + 50, len(text))].lower()
                dur = int(num) * 12 if 'tahun' in surrounding else int(num)
                if dur > max_bulan:
                    max_bulan = dur

    # Klasifikasi berdasarkan total bulan
    # Threshold: 24 bulan (2 tahun) = median data & dasar hukum KUHP
    if max_bulan == 0:
        return "tidak_diketahui"
    elif max_bulan <= 24:
        return "ringan"
    else:
        return "berat"


# -------------------------------------------------------
# FUNGSI 6: Pipeline utama representasi kasus
# -------------------------------------------------------
def build_case_representation(cleaned_csv_path, output_path):
    """
    Pipeline utama: membaca cleaned_cases.csv dan menghasilkan cases.csv
    dengan representasi lengkap setiap kasus.

    Parameter:
        cleaned_csv_path (str): Path ke cleaned_cases.csv
        output_path (str): Path untuk menyimpan cases.csv

    Return:
        pd.DataFrame: DataFrame representasi kasus
    """
    logger.info(f"Membaca data dari: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path, encoding='utf-8-sig')

    logger.info(f"Memproses {len(df)} kasus...")
    records = []

    for _, row in df.iterrows():
        case_id = row['case_id']
        text = str(row['cleaned_text'])

        # Ekstrak metadata
        meta = extract_metadata(text)

        # Ekstrak fitur statistik
        features = extract_text_features(text)

        # Buat ringkasan
        ringkasan = simple_textrank_summary(text, n_sentences=3)

        # Ekstrak amar putusan
        amar = extract_amar_putusan(text)

        # Klasifikasi hukuman
        kategori = classify_hukuman(text)

        record = {
            "case_id": case_id,
            "nomor_perkara": meta["nomor_perkara"],
            "tanggal_putusan": meta["tanggal_putusan"],
            "pasal": meta["pasal"],
            "nama_terdakwa": meta["nama_terdakwa"],
            "nama_hakim": meta["nama_hakim"],
            "jumlah_kata": features["jumlah_kata"],
            "jumlah_kalimat": features["jumlah_kalimat"],
            "panjang_dokumen": features["panjang_dokumen"],
            "ringkasan_fakta": ringkasan,
            "amar_putusan": amar,
            "kategori_hukuman": kategori,
            "cleaned_text": text
        }
        records.append(record)

    result_df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Representasi kasus tersimpan: {output_path}")

    return result_df


if __name__ == "__main__":
    df = build_case_representation(
        "data/processed/cleaned_cases.csv",
        "data/processed/cases.csv"
    )
    print(df[['case_id', 'pasal', 'kategori_hukuman', 'jumlah_kata']].head(10))
    print(f"\nDistribusi kategori hukuman:\n{df['kategori_hukuman'].value_counts()}")
