# =====================================================
# retrieval.py - Stage 3: Case Retrieval
# Sistem CBR Putusan Pengadilan - Pidana Penggelapan
# =====================================================
# Pendekatan: TF-IDF + Cosine Similarity + SVM
# =====================================================

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# FUNGSI 1: Melatih TF-IDF Vectorizer
# -------------------------------------------------------
def train_tfidf(texts, max_features=3000):
    """
    Melatih TF-IDF vectorizer pada kumpulan teks.

    TF-IDF (Term Frequency - Inverse Document Frequency):
    - TF: seberapa sering kata muncul dalam dokumen
    - IDF: seberapa jarang kata muncul di seluruh dokumen
    - Nilai tinggi = kata penting dan unik dalam dokumen tsb

    Parameter:
        texts (list): Daftar teks dokumen
        max_features (int): Maksimum fitur/kata (3000 optimal untuk ~80-100 dokumen)

    Return:
        TfidfVectorizer: Vectorizer yang sudah dilatih
        scipy.sparse.matrix: Matriks TF-IDF
    """
    logger.info(f"Melatih TF-IDF dengan max_features={max_features}...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # unigram dan bigram
        min_df=2,                 # kata harus muncul minimal di 2 dokumen
        max_df=0.95,              # abaikan kata yang ada di > 95% dokumen
        sublinear_tf=True,        # gunakan log normalisasi
        strip_accents='unicode',
        analyzer='word'
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


# -------------------------------------------------------
# FUNGSI 2: Melatih model SVM
# -------------------------------------------------------
def train_svm(X_train, y_train):
    """
    Melatih model SVM (Support Vector Machine) untuk klasifikasi kategori hukuman.

    SVM bekerja dengan mencari hyperplane optimal yang memisahkan kelas-kelas.
    Cocok untuk data teks berdimensi tinggi (seperti TF-IDF).

    Parameter:
        X_train: Matriks fitur training
        y_train: Label kategori hukuman

    Return:
        SVC: Model SVM yang sudah dilatih
    """
    logger.info("Melatih model SVM...")

    from sklearn.calibration import CalibratedClassifierCV

    base_model = SVC(
        kernel='linear',     # kernel linear cocok untuk teks
        C=1.0,               # regularization parameter
        class_weight='balanced',  # atasi class imbalance
        random_state=42
    )
    model = CalibratedClassifierCV(base_model, ensemble=False)

    try:
        model.fit(X_train, y_train)
        logger.info("Model SVM berhasil dilatih!")
        return model
    except ValueError as e:
        if "The number of classes has to be greater than one" in str(e) or "got 1 class" in str(e):
            raise ValueError(
                "Data training hanya memiliki 1 kelas. "
                "Sistem memerlukan minimal 2 kelas berbeda untuk melatih model SVM. "
                "Pastikan data mencakup variasi kategori hukuman (ringan/berat)."
            ) from e
        raise e


# -------------------------------------------------------
# FUNGSI 3: Fungsi utama retrieve
# -------------------------------------------------------
def retrieve(query, vectorizer, tfidf_matrix, df_cases, top_k=5, exclude_case_id=None):
    """
    Mengambil kasus paling mirip dengan query menggunakan Cosine Similarity.

    Cosine Similarity mengukur sudut antara dua vektor TF-IDF:
    - Nilai 1.0 = sangat mirip (sudut 0°)
    - Nilai 0.0 = tidak mirip sama sekali (sudut 90°)

    Parameter:
        query (str): Teks kasus baru yang ingin dicari kemiripannya
        vectorizer: TF-IDF vectorizer yang sudah dilatih
        tfidf_matrix: Matriks TF-IDF dari case base
        df_cases (pd.DataFrame): DataFrame kasus
        top_k (int): Jumlah kasus paling mirip yang diambil
        exclude_case_id (str or None): ID kasus yang dikecualikan (untuk evaluasi,
            agar kasus test tidak mengambil dirinya sendiri dari case base)

    Return:
        pd.DataFrame: top_k kasus dengan kolom [case_id, similarity_score, ...]
    """
    # Ubah query menjadi vektor TF-IDF
    query_vector = vectorizer.transform([query])

    # Hitung cosine similarity antara query dengan semua kasus
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Buat pasangan (similarity, index) dan urutkan dari tertinggi
    scored_indices = list(enumerate(similarities))
    scored_indices.sort(key=lambda x: x[1], reverse=True)

    # Ambil top_k, skip yang case_id-nya sama dengan exclude_case_id
    top_indices = []
    for idx, score in scored_indices:
        if exclude_case_id is not None and 'case_id' in df_cases.columns:
            if df_cases.iloc[idx]['case_id'] == exclude_case_id:
                continue
        top_indices.append(idx)
        if len(top_indices) >= top_k:
            break

    # Buat DataFrame hasil
    results = []
    for idx in top_indices:
        row = df_cases.iloc[idx].to_dict()
        row['similarity_score'] = round(float(similarities[idx]), 4)
        results.append(row)

    result_df = pd.DataFrame(results)

    # Tampilkan hanya kolom penting
    cols_to_show = ['case_id', 'similarity_score', 'pasal',
                    'kategori_hukuman', 'amar_putusan']
    cols_available = [c for c in cols_to_show if c in result_df.columns]

    logger.info(f"Ditemukan {len(result_df)} kasus paling mirip")
    return result_df[cols_available + ['cleaned_text']]


# -------------------------------------------------------
# FUNGSI 3B: Hybrid KNN Classifier (Cosine + Euclidean)
# -------------------------------------------------------
def hybrid_knn_predict(query_vector, tfidf_matrix, labels, k=7, alpha=0.6):
    """
    KNN classifier yang menggabungkan Cosine Similarity dan Euclidean Distance.

    Cosine bagus untuk arah/semantik teks, Euclidean bagus untuk magnitudo/panjang.
    Kombinasi keduanya lebih robust dari salah satu saja.

    Parameter:
        query_vector: Vektor TF-IDF query (sparse matrix)
        tfidf_matrix: Matriks TF-IDF case base
        labels: Array label kategori hukuman
        k (int): Jumlah nearest neighbors (default 7, lebih stabil dari 5)
        alpha (float): Bobot cosine (0-1). 0.6 = 60% cosine + 40% euclidean

    Return:
        str: Prediksi kategori hukuman
        float: Confidence score (0-1)
    """
    from scipy.spatial.distance import cdist

    # Cosine similarity
    cos_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Euclidean distance (efisien untuk sparse matrix)
    q_dense = query_vector.toarray()
    c_dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else tfidf_matrix
    euc_dist = cdist(q_dense, c_dense, metric='euclidean').flatten()
    max_dist = euc_dist.max() if euc_dist.max() > 0 else 1
    euc_sim = 1 - (euc_dist / max_dist)

    # Hybrid score
    hybrid_score = alpha * cos_sim + (1 - alpha) * euc_sim

    # Ambil top-k
    top_indices = hybrid_score.argsort()[::-1][:k]

    # Weighted voting
    votes = {}
    for idx in top_indices:
        label = labels[idx]
        score = float(hybrid_score[idx])
        votes[label] = votes.get(label, 0) + score

    prediksi = max(votes, key=votes.get)
    total = sum(votes.values())
    confidence = votes[prediksi] / total if total > 0 else 0

    return prediksi, round(confidence, 4)


# -------------------------------------------------------
# FUNGSI 4: Pipeline lengkap training dan simpan model
# -------------------------------------------------------
def build_retrieval_system(cases_csv_path, model_dir="models"):
    """
    Pipeline utama membangun sistem retrieval:
    1. Load data
    2. Split train/test (80/20)
    3. Latih TF-IDF
    4. Latih SVM
    5. Simpan model

    Parameter:
        cases_csv_path (str): Path ke cases.csv
        model_dir (str): Folder untuk menyimpan model

    Return:
        dict: Berisi vectorizer, model_svm, tfidf_matrix,
              df_train, df_test, label_encoder
    """
    logger.info(f"Loading data dari: {cases_csv_path}")
    df = pd.read_csv(cases_csv_path, encoding='utf-8-sig')

    # Filter baris dengan teks yang valid
    df = df.dropna(subset=['cleaned_text', 'kategori_hukuman'])
    df = df[df['kategori_hukuman'] != 'tidak_diketahui'].reset_index(drop=True)

    if len(df) < 10:
        logger.warning(f"Data hanya memiliki {len(df)} kasus valid. Disarankan minimal 10 kasus.")
        if len(df) < 2:
            raise ValueError(
                f"Data terlalu sedikit ({len(df)} kasus). "
                "Sistem memerlukan minimal 2 kasus dengan kategori hukuman berbeda "
                "untuk membangun model retrieval. Pastikan cases.csv memiliki data yang cukup."
            )

    logger.info(f"Total kasus valid: {len(df)}")

    # Encode label
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['kategori_hukuman'])

    # Cek apakah bisa menggunakan stratify (minimal 2 sampel per kelas)
    label_counts = df['label'].value_counts()
    can_stratify = label_counts.min() >= 2

    # Split data 80% train, 20% test
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df['label'] if can_stratify else None
    )
    logger.info(f"Train: {len(df_train)} | Test: {len(df_test)}")

    # Latih TF-IDF pada data training
    vectorizer, X_train = train_tfidf(df_train['cleaned_text'].tolist())

    # Transform data test
    X_test = vectorizer.transform(df_test['cleaned_text'].tolist())

    # Latih SVM
    svm_model = train_svm(X_train, df_train['label'])

    # Hitung TF-IDF untuk SEMUA data (untuk retrieval)
    tfidf_all = vectorizer.transform(df['cleaned_text'].tolist())

    # Simpan model
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, model_path / "tfidf_vectorizer.pkl")
    joblib.dump(svm_model, model_path / "svm_model.pkl")
    joblib.dump(le, model_path / "label_encoder.pkl")
    joblib.dump(tfidf_all, model_path / "tfidf_matrix.pkl")

    logger.info(f"Model tersimpan di folder: {model_dir}/")

    return {
        "vectorizer": vectorizer,
        "svm_model": svm_model,
        "tfidf_matrix": tfidf_all,
        "label_encoder": le,
        "df_all": df,
        "df_train": df_train,
        "df_test": df_test,
        "X_train": X_train,
        "X_test": X_test
    }


def load_retrieval_system(cases_csv_path, model_dir="models"):
    """
    Memuat model yang sudah dilatih dari disk.

    Return: dict dengan vectorizer, model_svm, tfidf_matrix, df_cases, label_encoder
    """
    model_path = Path(model_dir)
    vectorizer = joblib.load(model_path / "tfidf_vectorizer.pkl")
    svm_model = joblib.load(model_path / "svm_model.pkl")
    le = joblib.load(model_path / "label_encoder.pkl")
    tfidf_matrix = joblib.load(model_path / "tfidf_matrix.pkl")
    df = pd.read_csv(cases_csv_path, encoding='utf-8-sig')

    logger.info("Model berhasil dimuat dari disk")
    return {
        "vectorizer": vectorizer,
        "svm_model": svm_model,
        "tfidf_matrix": tfidf_matrix,
        "label_encoder": le,
        "df_all": df
    }


if __name__ == "__main__":
    system = build_retrieval_system("data/processed/cases.csv")
    query_contoh = "terdakwa menggelapkan uang perusahaan sebesar rp 50 juta sebagai bendahara"
    results = retrieve(
        query_contoh,
        system["vectorizer"],
        system["tfidf_matrix"],
        system["df_all"],
        top_k=5
    )
    print("\nHasil Retrieval:")
    print(results[['case_id', 'similarity_score', 'kategori_hukuman']])
