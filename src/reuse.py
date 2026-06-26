# =====================================================
# reuse.py - Stage 4: Case Solution Reuse
# Sistem CBR Putusan Pengadilan - Pidana Penggelapan
# =====================================================
# Pendekatan: Majority Voting + Weighted Similarity Voting
# =====================================================

import logging
import numpy as np
import pandas as pd
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# FUNGSI 1: Majority Voting
# -------------------------------------------------------
def majority_voting(retrieved_cases):
    """
    Metode 1: Majority Voting (Pemungutan Suara Mayoritas)

    Cara kerja:
    - Ambil top-k kasus mirip
    - Tiap kasus 'memberi suara' untuk kategori hukumannya
    - Kategori dengan suara terbanyak = prediksi

    Kelebihan: Sederhana, mudah dipahami
    Kekurangan: Semua kasus dianggap sama bobot-nya

    Parameter:
        retrieved_cases (pd.DataFrame): Hasil fungsi retrieve() dengan kolom
            [case_id, similarity_score, kategori_hukuman]

    Return:
        str: Prediksi kategori hukuman
        dict: Detail voting
    """
    if 'kategori_hukuman' not in retrieved_cases.columns:
        return "tidak_diketahui", {}

    # Hitung suara untuk tiap kategori
    votes = Counter(retrieved_cases['kategori_hukuman'].tolist())

    # Prediksi = kategori dengan suara terbanyak
    prediksi = votes.most_common(1)[0][0]

    logger.info(f"Majority Voting: {dict(votes)} -> Prediksi: {prediksi}")
    return prediksi, dict(votes)


# -------------------------------------------------------
# FUNGSI 2: Weighted Similarity Voting
# -------------------------------------------------------
def weighted_voting(retrieved_cases):
    """
    Metode 2: Weighted Similarity Voting (Voting Berbobot)

    Cara kerja:
    - Ambil top-k kasus mirip dengan similarity score-nya
    - Tiap kasus memberi suara dengan BOBOT = similarity score-nya
    - Kasus yang lebih mirip = suara lebih berat
    - Kategori dengan total bobot tertinggi = prediksi

    Kelebihan: Lebih akurat karena mempertimbangkan tingkat kemiripan
    Kekurangan: Lebih kompleks dari majority voting

    Parameter:
        retrieved_cases (pd.DataFrame): Hasil retrieve() dengan kolom
            [case_id, similarity_score, kategori_hukuman]

    Return:
        str: Prediksi kategori hukuman
        dict: Detail bobot tiap kategori
    """
    if 'kategori_hukuman' not in retrieved_cases.columns:
        return "tidak_diketahui", {}

    # Hitung bobot per kategori
    weighted_scores = {}
    for _, row in retrieved_cases.iterrows():
        kategori = row.get('kategori_hukuman', 'tidak_diketahui')
        skor = float(row.get('similarity_score', 0))
        weighted_scores[kategori] = weighted_scores.get(kategori, 0) + skor

    # Prediksi = kategori dengan total bobot tertinggi
    prediksi = max(weighted_scores, key=weighted_scores.get)

    # Bulatkan untuk tampilan
    weighted_scores = {k: round(v, 4) for k, v in weighted_scores.items()}

    logger.info(f"Weighted Voting: {weighted_scores} -> Prediksi: {prediksi}")
    return prediksi, weighted_scores


# -------------------------------------------------------
# FUNGSI 2B: Power Weighted Voting (enhanced)
# -------------------------------------------------------
def power_weighted_voting(retrieved_cases, power=2.0):
    """
    Metode 2B: Power Weighted Voting - peningkatan dari weighted voting.

    Cara kerja:
    - Sama seperti weighted voting, tapi bobot similarity dipangkatkan (power)
    - Power > 1: memperbesar perbedaan antara kasus sangat mirip vs kurang mirip
    - Power < 1: memperhalus perbedaan (lebih demokratis)
    - Default power=2.0: kasus 2x lebih mirip mendapat 4x bobot

    Parameter:
        retrieved_cases (pd.DataFrame): Hasil retrieve() dengan kolom
            [case_id, similarity_score, kategori_hukuman]
        power (float): Pangkat untuk amplifikasi bobot (default 2.0)

    Return:
        str: Prediksi kategori hukuman
        dict: Detail bobot tiap kategori
    """
    if 'kategori_hukuman' not in retrieved_cases.columns:
        return "tidak_diketahui", {}

    weighted_scores = {}
    for _, row in retrieved_cases.iterrows():
        kategori = row.get('kategori_hukuman', 'tidak_diketahui')
        skor = float(row.get('similarity_score', 0))
        bobot = skor ** power  # amplifikasi perbedaan similarity
        weighted_scores[kategori] = weighted_scores.get(kategori, 0) + bobot

    prediksi = max(weighted_scores, key=weighted_scores.get)
    weighted_scores = {k: round(v, 4) for k, v in weighted_scores.items()}

    logger.info(f"Power Weighted (p={power}): {weighted_scores} -> Prediksi: {prediksi}")
    return prediksi, weighted_scores


# -------------------------------------------------------
# FUNGSI 2C: Hybrid Ensemble (SVM + Weighted Voting + KNN)
# -------------------------------------------------------
def hybrid_ensemble(svm_pred, svm_proba, weighted_pred, weighted_scores,
                     knn_pred=None, knn_confidence=None):
    """
    Metode ensemble yang menggabungkan 3 sumber prediksi:
    1. SVM (model klasifikasi berbasis fitur TF-IDF)
    2. Weighted Voting (berdasarkan similaritas kasus)
    3. Hybrid KNN (cosine + euclidean, opsional)

    Setiap sumber memberi kontribusi berdasarkan confidence-nya.

    Parameter:
        svm_pred (str): Prediksi SVM
        svm_proba (float): Probabilitas SVM (0-1)
        weighted_pred (str): Prediksi weighted voting
        weighted_scores (dict): Skor weighted voting per kategori
        knn_pred (str or None): Prediksi hybrid KNN (opsional)
        knn_confidence (float or None): Confidence KNN (0-1, opsional)

    Return:
        str: Prediksi final ensemble
        dict: Detail skor ensemble per kategori
    """
    # Bobot untuk setiap sumber prediksi
    W_SVM = 0.40
    W_VOTE = 0.35
    W_KNN = 0.25 if knn_pred is not None else 0.0

    # Normalisasi jika KNN tidak ada
    if knn_pred is None:
        total_w = W_SVM + W_VOTE
        W_SVM /= total_w
        W_VOTE /= total_w

    # Kumpulkan semua kategori
    all_categories = set([svm_pred, weighted_pred])
    if knn_pred is not None:
        all_categories.add(knn_pred)
    all_categories.update(weighted_scores.keys())

    # Hitung skor ensemble per kategori
    ensemble_scores = {}
    for kategori in all_categories:
        skor = 0.0

        # Kontribusi SVM
        if kategori == svm_pred:
            skor += W_SVM * svm_proba
        else:
            skor += W_SVM * (1 - svm_proba) / max(len(all_categories) - 1, 1)

        # Kontribusi Weighted Voting (normalisasi skor)
        total_vote = sum(weighted_scores.values()) if weighted_scores else 1
        vote_share = weighted_scores.get(kategori, 0) / total_vote if total_vote > 0 else 0
        skor += W_VOTE * vote_share

        # Kontribusi KNN (jika ada)
        if knn_pred is not None and knn_confidence is not None:
            if kategori == knn_pred:
                skor += W_KNN * knn_confidence
            else:
                skor += W_KNN * (1 - knn_confidence) / max(len(all_categories) - 1, 1)

        ensemble_scores[kategori] = round(skor, 4)

    prediksi = max(ensemble_scores, key=ensemble_scores.get)
    logger.info(f"Hybrid Ensemble: {ensemble_scores} -> Prediksi: {prediksi}")
    return prediksi, ensemble_scores


# -------------------------------------------------------
# FUNGSI 3: Fungsi utama predict_outcome
# -------------------------------------------------------
def predict_outcome(query, vectorizer, tfidf_matrix, df_cases, top_k=15,
                     svm_model=None, label_encoder=None):
    """
    Fungsi utama CBR: Prediksi hasil putusan untuk kasus baru.

    Siklus CBR yang diimplementasikan:
    RETRIEVE -> REUSE (dengan 4 metode: Majority, Weighted, Power Weighted, Hybrid Ensemble)

    Parameter:
        query (str): Deskripsi kasus baru
        vectorizer: TF-IDF vectorizer
        tfidf_matrix: Matriks TF-IDF case base
        df_cases (pd.DataFrame): DataFrame semua kasus
        top_k (int): Jumlah kasus mirip yang diambil (default 15, optimal untuk dataset putusan)
        svm_model: Model SVM (opsional, untuk hybrid ensemble)
        label_encoder: LabelEncoder (opsional, untuk hybrid ensemble)

    Return:
        dict: Hasil prediksi semua metode
    """
    from retrieval import retrieve, hybrid_knn_predict

    logger.info("=" * 50)
    logger.info("PREDIKSI HASIL KASUS BARU")
    logger.info(f"Query: {query[:100]}...")
    logger.info("=" * 50)

    # Step 1: RETRIEVE - Ambil kasus paling mirip
    retrieved = retrieve(query, vectorizer, tfidf_matrix, df_cases, top_k=top_k)
    logger.info(f"\nTop {top_k} kasus mirip berhasil diambil")

    # Step 2: REUSE - Majority Voting
    majority_pred, majority_detail = majority_voting(retrieved)

    # Step 3: REUSE - Weighted Voting
    weighted_pred, weighted_detail = weighted_voting(retrieved)

    # Step 4: REUSE - Power Weighted Voting (enhanced)
    power_pred, power_detail = power_weighted_voting(retrieved, power=2.0)

    # Step 5: Hybrid Ensemble (jika SVM tersedia)
    hybrid_pred = None
    hybrid_detail = {}
    if svm_model is not None and label_encoder is not None:
        query_vec = vectorizer.transform([query])

        # SVM prediction
        svm_label = svm_model.predict(query_vec)[0]
        svm_pred = label_encoder.inverse_transform([svm_label])[0]
        try:
            svm_proba = max(svm_model.predict_proba(query_vec)[0])
        except Exception:
            svm_proba = 0.6  # default confidence

        # Hybrid KNN prediction
        knn_pred, knn_conf = hybrid_knn_predict(
            query_vec, tfidf_matrix, df_cases['kategori_hukuman'].values, k=15
        )

        hybrid_pred, hybrid_detail = hybrid_ensemble(
            svm_pred, svm_proba,
            weighted_pred, weighted_detail,
            knn_pred, knn_conf
        )

    # Tampilkan perbandingan
    logger.info(f"\nHasil Perbandingan:")
    logger.info(f"  Majority Voting  : {majority_pred}")
    logger.info(f"  Weighted Voting  : {weighted_pred}")
    logger.info(f"  Power Weighted   : {power_pred}")
    if hybrid_pred:
        logger.info(f"  Hybrid Ensemble  : {hybrid_pred}")

    return {
        "retrieved_cases": retrieved,
        "majority_result": majority_pred,
        "majority_detail": majority_detail,
        "weighted_result": weighted_pred,
        "weighted_detail": weighted_detail,
        "power_weighted_result": power_pred,
        "power_weighted_detail": power_detail,
        "hybrid_result": hybrid_pred,
        "hybrid_detail": hybrid_detail,
        "consistent": majority_pred == weighted_pred == power_pred
    }


# -------------------------------------------------------
# FUNGSI 4: Format hasil prediksi untuk tampilan
# -------------------------------------------------------
def format_prediction_result(result):
    """
    Memformat hasil prediksi menjadi string yang mudah dibaca.

    Parameter:
        result (dict): Output dari predict_outcome()

    Return:
        str: Hasil yang sudah diformat
    """
    output = []
    output.append("=" * 60)
    output.append("HASIL PREDIKSI SISTEM CBR")
    output.append("=" * 60)

    output.append("\n📋 KASUS MIRIP YANG DITEMUKAN:")
    df = result['retrieved_cases']
    for i, row in df.iterrows():
        output.append(
            f"  {i+1}. [{row.get('case_id', '?')}] "
            f"Similarity: {row.get('similarity_score', 0):.4f} | "
            f"Kategori: {row.get('kategori_hukuman', '?')}"
        )

    output.append("\n🗳️  METODE 1 - MAJORITY VOTING:")
    output.append(f"  Detail Suara : {result['majority_detail']}")
    output.append(f"  Prediksi     : ⚖️  {result['majority_result'].upper()}")

    output.append("\n⚖️  METODE 2 - WEIGHTED SIMILARITY VOTING:")
    output.append(f"  Detail Bobot : {result['weighted_detail']}")
    output.append(f"  Prediksi     : ⚖️  {result['weighted_result'].upper()}")

    konsisten = "✅ YA - Kedua metode sepakat" if result['consistent'] else "⚠️  TIDAK - Hasil berbeda"
    output.append(f"\n🔍 KONSISTENSI: {konsisten}")
    output.append("=" * 60)

    return "\n".join(output)


if __name__ == "__main__":
    # Contoh penggunaan (perlu sistem retrieval)
    print("Modul reuse siap digunakan.")
    print("Gunakan fungsi predict_outcome() dengan vectorizer dan tfidf_matrix dari retrieval.py")
