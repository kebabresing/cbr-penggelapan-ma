# Sistem Case-Based Reasoning (CBR)
# Analisis Putusan Pengadilan Indonesia
## Domain: Pidana Umum Penggelapan

> **Tugas Kuliah Penalaran Komputer Sub CPMK-3**  
> **Nama & NIM:**  
> - Akhmad Zamri Ardani (202310370311406)  
> - Achmad Rizqy Nur (202310370311430)  
> **Mata Kuliah:** Penalaran Komputer  
> Dataset: Direktori Putusan Mahkamah Agung RI  
> Bahasa: Python 3.11+


---

## Deskripsi Proyek

Proyek ini membangun sistem **Case-Based Reasoning (CBR)** untuk menganalisis putusan pengadilan Indonesia dalam domain **Pidana Umum Penggelapan**. Sistem menggunakan pendekatan:

- **TF-IDF** untuk representasi teks
- **Cosine Similarity** untuk pengukuran kemiripan
- **SVM** sebagai classifier
- **Majority Voting, Weighted Voting & Power Weighted Voting** untuk prediksi solusi

---

## Siklus CBR yang Diimplementasikan

```
┌─────────────────────────────────────────────────────────┐
│                   SIKLUS CBR                            │
│                                                         │
│  Kasus Baru                                             │
│      │                                                  │
│      ▼                                                  │
│  ┌──────────────────┐                                   │
│  │  1. CASE BASE    │  ← PDF Putusan MA → cleaned_cases │
│  │  CONSTRUCTION    │                                   │
│  └────────┬─────────┘                                   │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────┐                                   │
│  │  2. CASE         │  ← Metadata, Fitur, Ringkasan     │
│  │  REPRESENTATION  │    → cases.csv                    │
│  └────────┬─────────┘                                   │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────┐                                   │
│  │  3. CASE         │  ← TF-IDF + Cosine Similarity     │
│  │  RETRIEVAL       │    → Top-K kasus mirip            │
│  └────────┬─────────┘                                   │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────┐                                   │
│  │  4. CASE REUSE   │  ← Majority + Weighted + Power Voting │
│  │  (Solution)      │    → Prediksi kategori hukuman    │
│  └────────┬─────────┘                                   │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────┐                                   │
│  │  5. EVALUATION   │  ← Accuracy, Precision, F1        │
│  │                  │    → Confusion Matrix, Report     │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Struktur Proyek

```
project/
├── data/
│   ├── raw/              ← Letakkan file PDF di sini
│   ├── processed/        ← cleaned_cases.csv, cases.csv
│   ├── eval/             ← retrieval_metrics.csv, prediction_metrics.csv
│   └── results/          ← Hasil akhir
│
├── notebooks/
│   ├── 01_case_base.ipynb          ← Stage 1: Baca & bersihkan PDF
│   ├── 02_case_representation.ipynb ← Stage 2: Ekstraksi fitur & metadata
│   ├── 03_case_retrieval.ipynb     ← Stage 3: TF-IDF + SVM + Retrieval
│   ├── 04_case_solution_reuse.ipynb ← Stage 4: Voting & Prediksi
│   └── 05_evaluation.ipynb         ← Stage 5: Evaluasi & Visualisasi
│
├── src/
│   ├── preprocessing.py    ← Modul baca PDF & cleaning teks
│   ├── representation.py   ← Modul ekstraksi metadata & fitur
│   ├── retrieval.py        ← Modul TF-IDF + Cosine Similarity + SVM
│   ├── reuse.py            ← Modul Majority, Weighted & Power Voting
│   └── evaluation.py       ← Modul metrik & visualisasi
│
├── models/                 ← Model tersimpan (.pkl)
├── outputs/                ← Gambar visualisasi
├── test_pipeline.py        ← Script eksekusi pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Instalasi Dependencies

```bash
pip install -r requirements.txt
```

### 2. Siapkan Data PDF

Letakkan file PDF putusan pengadilan ke folder `data/raw/`.  
Folder `data/raw/` harus berisi file PDF putusan pengadilan agar sistem dapat memproses data.

### 3. Jalankan Pipeline End-to-End (Cepat)

```bash
python test_pipeline.py
```

Script ini menjalankan seluruh pipeline secara otomatis: PDF extraction → representasi → retrieval → prediksi → evaluasi.

### 4. Jalankan Notebook Secara Berurutan (Demo)

```bash
cd notebooks
jupyter notebook
```

Jalankan notebook dalam urutan:
1. `01_case_base.ipynb`
2. `02_case_representation.ipynb`
3. `03_case_retrieval.ipynb`
4. `04_case_solution_reuse.ipynb`
5. `05_evaluation.ipynb`

> **Catatan:** Setiap notebook dapat dijalankan secara **independen** selama data dari notebook sebelumnya sudah ada.

---

## Cara Download Dataset dari Direktori Putusan MA

1. Buka: https://putusan3.mahkamahagung.go.id/
2. Pilih **Direktori Putusan**
3. Filter:
   - Klasifikasi: **Pidana**
   - Kata kunci: **penggelapan**
4. Download minimal **50 putusan** dalam format PDF
5. Simpan ke folder `data/raw/`

---

## Penggunaan Modul Secara Langsung

### Retrieval (Cari Kasus Mirip)

```python
import sys
sys.path.append('src')

import joblib
import pandas as pd
from retrieval import retrieve

# Load model
vectorizer   = joblib.load('models/tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('models/tfidf_matrix.pkl')
df_cases     = pd.read_csv('data/processed/cases.csv', encoding='utf-8-sig')

# Cari kasus mirip
query = "terdakwa bendahara menggelapkan uang rp 50 juta pasal 374 kuhp"
hasil = retrieve(query, vectorizer, tfidf_matrix, df_cases, top_k=15)
print(hasil[['case_id', 'similarity_score', 'kategori_hukuman']])
```

### Prediksi Hukuman

```python
from reuse import predict_outcome, format_prediction_result

result = predict_outcome(
    query=query,
    vectorizer=vectorizer,
    tfidf_matrix=tfidf_matrix,
    df_cases=df_cases,
    top_k=15
)
print(format_prediction_result(result))
```

---

## Metrik Evaluasi

| Metrik | Deskripsi |
|--------|-----------|
| **Accuracy** | % prediksi benar dari total prediksi |
| **Precision** | % prediksi positif yang benar |
| **Recall** | % data positif yang berhasil diprediksi |
| **F1-Score** | Rata-rata harmonik Precision & Recall |
| **Precision@K** | Kualitas top-K retrieval |

---

## Kategori Hukuman

| Kategori | Rentang Hukuman |
|----------|----------------|
| **Ringan** | ≤ 24 bulan (2 tahun) |
| **Berat** | > 24 bulan (> 2 tahun) |

---

## Tech Stack

- **Python 3.11+**
- **pdfplumber** - Ekstraksi teks dari PDF
- **pandas / numpy** - Manipulasi data
- **scikit-learn** - TF-IDF, SVM, metrik evaluasi
- **matplotlib / seaborn** - Visualisasi
- **joblib** - Simpan & load model
- **tqdm** - Progress bar

---

## Catatan Pengembangan

- Kode dirancang untuk **kemudahan pemahaman mahasiswa**
- Setiap fungsi memiliki **docstring** yang menjelaskan konsep CBR
- Sistem memerlukan data PDF asli untuk menghasilkan model yang akurat
- Setiap notebook dapat dijalankan **secara independen**

---

## Referensi

1. Aamodt, A., & Plaza, E. (1994). Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches.
2. Manning, C.D., et al. (2008). Introduction to Information Retrieval.
3. Direktori Putusan Mahkamah Agung RI: https://putusan3.mahkamahagung.go.id/

---

*Proyek ini dibuat untuk keperluan tugas kuliah Penalaran Komputer.*
