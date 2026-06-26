# =====================================================
# evaluation.py - Stage 5: Evaluation
# Sistem CBR Putusan Pengadilan - Pidana Penggelapan
# =====================================================

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# FUNGSI 1: Evaluasi model retrieval (SVM)
# -------------------------------------------------------
def evaluate_retrieval(svm_model, X_test, y_test, label_encoder, output_dir="data/eval"):
    """
    Mengevaluasi performa model SVM pada data test.

    Menghitung: Accuracy, Precision, Recall, F1-Score
    Menghasilkan: classification_report, confusion_matrix

    Parameter:
        svm_model: Model SVM yang sudah dilatih
        X_test: Matriks fitur test
        y_test: Label aktual (encoded)
        label_encoder: LabelEncoder untuk dekode label
        output_dir (str): Folder untuk menyimpan hasil

    Return:
        dict: Berisi semua metrik evaluasi
    """
    logger.info("Mengevaluasi model retrieval (SVM)...")

    # Prediksi
    y_pred = svm_model.predict(X_test)

    # Decode label ke nama kategori
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    logger.info(f"Accuracy  : {accuracy:.4f}")
    logger.info(f"Precision : {precision:.4f}")
    logger.info(f"Recall    : {recall:.4f}")
    logger.info(f"F1-Score  : {f1:.4f}")

    # Classification Report
    report = classification_report(
        y_test_labels, y_pred_labels,
        zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Simpan metrik ke CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([{
        "model": "SVM (TF-IDF)",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }])
    metrics_df.to_csv(f"{output_dir}/retrieval_metrics.csv", index=False, encoding='utf-8-sig')
    logger.info(f"Metrik tersimpan: {output_dir}/retrieval_metrics.csv")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "y_test_labels": y_test_labels,
        "y_pred_labels": y_pred_labels,
        "report": report
    }


# -------------------------------------------------------
# FUNGSI 2: Evaluasi metode prediksi (Voting)
# -------------------------------------------------------
def evaluate_predictions(y_true, y_pred_majority, y_pred_weighted,
                          output_dir="data/eval"):
    """
    Membandingkan performa dua metode prediksi:
    Majority Voting vs Weighted Similarity Voting

    Parameter:
        y_true (list): Label aktual
        y_pred_majority (list): Prediksi majority voting
        y_pred_weighted (list): Prediksi weighted voting
        output_dir (str): Folder output

    Return:
        dict: Metrik kedua metode
    """
    logger.info("Membandingkan metode prediksi...")

    def calc_metrics(y_true, y_pred, method_name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        logger.info(f"\n{method_name}: Acc={acc:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f}")
        return {"method": method_name, "accuracy": round(acc, 4),
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1_score": round(f1, 4)}

    majority_metrics = calc_metrics(y_true, y_pred_majority, "Majority Voting")
    weighted_metrics = calc_metrics(y_true, y_pred_weighted, "Weighted Voting")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame([majority_metrics, weighted_metrics])
    pred_df.to_csv(f"{output_dir}/prediction_metrics.csv", index=False, encoding='utf-8-sig')
    logger.info(f"Metrik prediksi tersimpan: {output_dir}/prediction_metrics.csv")

    return {
        "majority": majority_metrics,
        "weighted": weighted_metrics
    }


# -------------------------------------------------------
# FUNGSI 3: Visualisasi Confusion Matrix
# -------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                           output_path="outputs/confusion_matrix.png"):
    """
    Membuat dan menyimpan visualisasi confusion matrix.

    Confusion Matrix menunjukkan:
    - Baris = label aktual
    - Kolom = label prediksi
    - Diagonal = prediksi benar
    - Non-diagonal = prediksi salah
    """
    labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Label Aktual', fontsize=12)
    plt.xlabel('Label Prediksi', fontsize=12)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Confusion matrix tersimpan: {output_path}")


# -------------------------------------------------------
# FUNGSI 4: Visualisasi Perbandingan Metrik
# -------------------------------------------------------
def plot_metrics_comparison(metrics_dict, output_path="outputs/metrics_comparison.png"):
    """
    Membuat bar chart perbandingan metrik dua metode prediksi.

    Parameter:
        metrics_dict (dict): {
            'majority': {'accuracy': x, 'precision': y, ...},
            'weighted': {'accuracy': x, 'precision': y, ...}
        }
    """
    metrik_list = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrik_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    majority_vals = [metrics_dict['majority'].get(m, 0) for m in metrik_list]
    weighted_vals = [metrics_dict['weighted'].get(m, 0) for m in metrik_list]

    bars1 = ax.bar(x - width/2, majority_vals, width,
                   label='Majority Voting', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, weighted_vals, width,
                   label='Weighted Voting', color='coral', alpha=0.8)

    # Tambah label nilai di atas bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metrik Evaluasi', fontsize=12)
    ax.set_ylabel('Nilai', fontsize=12)
    ax.set_title('Perbandingan Metrik: Majority Voting vs Weighted Voting',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Grafik perbandingan tersimpan: {output_path}")


# -------------------------------------------------------
# FUNGSI 5: Evaluasi retrieval berbasis similarity
# -------------------------------------------------------
def evaluate_retrieval_similarity(df_test, vectorizer, tfidf_matrix, df_all, top_k=5):
    """
    Mengevaluasi kualitas retrieval menggunakan Precision@K.

    Precision@K = Berapa persen dari top-K kasus yang dikembalikan
    memiliki kategori sama dengan kasus query?

    Parameter:
        df_test: Data test
        vectorizer: TF-IDF vectorizer
        tfidf_matrix: Matriks TF-IDF
        df_all: Semua kasus
        top_k: Jumlah kasus yang diambil

    Return:
        float: Rata-rata Precision@K
    """
    from retrieval import retrieve

    logger.info(f"Menghitung Precision@{top_k}...")
    precisions = []

    for _, row in df_test.iterrows():
        query = str(row['cleaned_text'])
        true_label = row.get('kategori_hukuman', '')

        retrieved = retrieve(query, vectorizer, tfidf_matrix, df_all, top_k=top_k,
                              exclude_case_id=row.get('case_id'))

        if 'kategori_hukuman' not in retrieved.columns or len(retrieved) == 0:
            continue

        # Hitung berapa yang labelnya sama
        correct = (retrieved['kategori_hukuman'] == true_label).sum()
        precision_k = correct / len(retrieved)
        precisions.append(precision_k)

    avg_precision = np.mean(precisions) if precisions else 0
    logger.info(f"Rata-rata Precision@{top_k}: {avg_precision:.4f}")
    return avg_precision


if __name__ == "__main__":
    print("Modul evaluasi siap digunakan.")
    print("Gunakan fungsi evaluate_retrieval() dan evaluate_predictions()")
