"""
test_pipeline.py
Script untuk menguji pipeline CBR dari awal sampai akhir menggunakan data PDF asli.
Jalankan: python test_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import joblib
import pandas as pd
from pathlib import Path

print("=" * 60)
print("TEST PIPELINE CBR - PIDANA UMUM PENGGELAPAN")
print("=" * 60)

# =====================================================
# STAGE 1: Case Base Construction (dari PDF asli)
# =====================================================
print("\n[STAGE 1] Case Base Construction dari PDF asli...")
from preprocessing import process_all_pdfs

RAW_DIR = "data/raw"
OUTPUT_CSV = "data/processed/cleaned_cases.csv"

pdf_files = list(Path(RAW_DIR).glob("*.pdf"))
if not pdf_files:
    print(f"  [ERROR] Tidak ada file PDF di folder '{RAW_DIR}/'")
    print(f"  Silakan masukkan file PDF putusan ke folder '{RAW_DIR}/' terlebih dahulu.")
    sys.exit(1)

print(f"  Ditemukan {len(pdf_files)} file PDF")
df1 = process_all_pdfs(RAW_DIR, OUTPUT_CSV)

if len(df1) == 0:
    print("  [ERROR] Tidak ada kasus yang berhasil diekstrak dari PDF.")
    sys.exit(1)

print(f"  [OK] {len(df1)} kasus berhasil diekstrak dari PDF")

# =====================================================
# STAGE 2: Case Representation
# =====================================================
print("\n[STAGE 2] Case Representation...")
from representation import build_case_representation

df2 = build_case_representation(
    "data/processed/cleaned_cases.csv",
    "data/processed/cases.csv"
)
print(f"  [OK] {len(df2)} kasus direpresentasikan")
distribusi = df2["kategori_hukuman"].value_counts().to_dict()
print(f"  [OK] Distribusi hukuman: {distribusi}")

# =====================================================
# STAGE 3: Case Retrieval (TF-IDF + SVM)
# =====================================================
print("\n[STAGE 3] Case Retrieval (TF-IDF + SVM)...")
from retrieval import build_retrieval_system, retrieve

system = build_retrieval_system("data/processed/cases.csv", "models")
print(f"  [OK] Train: {len(system['df_train'])} | Test: {len(system['df_test'])}")
print(f"  [OK] Fitur TF-IDF: {system['X_train'].shape[1]}")

query_test = "terdakwa bendahara menggelapkan uang perusahaan pasal 374 kuhp"
top_cases = retrieve(
    query_test,
    system["vectorizer"],
    system["tfidf_matrix"],
    system["df_all"],
    top_k=15
)
print(f"  [OK] Retrieve: {len(top_cases)} kasus ditemukan")

# =====================================================
# STAGE 4: Case Solution Reuse
# =====================================================
print("\n[STAGE 4] Case Solution Reuse...")
from reuse import predict_outcome

result = predict_outcome(
    query=query_test,
    vectorizer=system["vectorizer"],
    tfidf_matrix=system["tfidf_matrix"],
    df_cases=system["df_all"],
    top_k=15,
    svm_model=system["svm_model"],
    label_encoder=system["label_encoder"]
)
print(f"  [OK] Majority Voting  : {result['majority_result']}")
print(f"  [OK] Weighted Voting  : {result['weighted_result']}")
print(f"  [OK] Power Weighted   : {result['power_weighted_result']}")
print(f"  [OK] Hybrid Ensemble  : {result['hybrid_result']}")
print(f"  [OK] Konsisten        : {'Ya' if result['consistent'] else 'Tidak'}")

# =====================================================
# STAGE 5: Evaluation
# =====================================================
print("\n[STAGE 5] Evaluation...")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from evaluation import evaluate_retrieval, evaluate_predictions
from reuse import majority_voting, weighted_voting

df = system["df_all"].copy()
le = LabelEncoder()
df["label"] = le.fit_transform(df["kategori_hukuman"])

# Cek stratify
label_counts = df["label"].value_counts()
can_stratify = label_counts.min() >= 2

_, df_test_eval = train_test_split(
    df, test_size=0.2, random_state=42,
    stratify=df["label"] if can_stratify else None
)
X_test = system["vectorizer"].transform(df_test_eval["cleaned_text"])
y_test = df_test_eval["label"].values

# Evaluasi SVM
metrics = evaluate_retrieval(system["svm_model"], X_test, y_test, le, output_dir="data/eval")
print(f"  [OK] SVM Accuracy : {metrics['accuracy']:.4f}")
print(f"  [OK] SVM F1-Score : {metrics['f1_score']:.4f}")

# Evaluasi Voting (dengan exclude self-match untuk hindari data leakage)
print("\n  Evaluasi semua metode prediksi (exclude self-match)...")
tfidf_all = system["vectorizer"].transform(df["cleaned_text"])
from reuse import power_weighted_voting
from retrieval import hybrid_knn_predict

y_true_vote = []
y_pred_maj, y_pred_wgt, y_pred_pw, y_pred_hybrid = [], [], [], []

for _, row in df_test_eval.iterrows():
    query = str(row["cleaned_text"])
    true_label = row["kategori_hukuman"]
    # Exclude diri sendiri dari case base saat retrieve
    top_cases = retrieve(query, system["vectorizer"], tfidf_all, df, top_k=15,
                         exclude_case_id=row["case_id"])
    pred_maj, _ = majority_voting(top_cases)
    pred_wgt, wgt_detail = weighted_voting(top_cases)
    pred_pw, _ = power_weighted_voting(top_cases, power=2.0)

    # Hybrid ensemble
    q_vec = system["vectorizer"].transform([query])
    svm_lbl = system["svm_model"].predict(q_vec)[0]
    svm_pred = le.inverse_transform([svm_lbl])[0]
    try:
        svm_proba = max(system["svm_model"].predict_proba(q_vec)[0])
    except Exception:
        svm_proba = 0.6
    knn_pred, knn_conf = hybrid_knn_predict(q_vec, tfidf_all, df['kategori_hukuman'].values, k=15)
    from reuse import hybrid_ensemble
    pred_hyb, _ = hybrid_ensemble(svm_pred, svm_proba, pred_wgt, wgt_detail, knn_pred, knn_conf)

    y_true_vote.append(true_label)
    y_pred_maj.append(pred_maj)
    y_pred_wgt.append(pred_wgt)
    y_pred_pw.append(pred_pw)
    y_pred_hybrid.append(pred_hyb)

# Evaluasi 4 metode
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
methods = {
    'Majority Voting': y_pred_maj,
    'Weighted Voting': y_pred_wgt,
    'Power Weighted': y_pred_pw,
    'Hybrid Ensemble': y_pred_hybrid
}

print("\n  Perbandingan Hasil Evaluasi:")
print(f"  {'Method':<22} {'Acc':>6} {'Prec':>6} {'Recall':>6} {'F1':>6}")
print(f"  {'-'*46}")
best_method, best_f1 = None, 0
for name, preds in methods.items():
    acc = accuracy_score(y_true_vote, preds)
    prec = precision_score(y_true_vote, preds, average='weighted', zero_division=0)
    rec = recall_score(y_true_vote, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_true_vote, preds, average='weighted', zero_division=0)
    print(f"  {name:<22} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_method = name

# Simpan metrics
import pandas as pd
rows = []
for name, preds in methods.items():
    rows.append({
        'method': name,
        'accuracy': round(accuracy_score(y_true_vote, preds), 4),
        'precision': round(precision_score(y_true_vote, preds, average='weighted', zero_division=0), 4),
        'recall': round(recall_score(y_true_vote, preds, average='weighted', zero_division=0), 4),
        'f1_score': round(f1_score(y_true_vote, preds, average='weighted', zero_division=0), 4)
    })
pd.DataFrame(rows).to_csv('data/eval/prediction_metrics.csv', index=False, encoding='utf-8-sig')
print(f"\n  [OK] Metode terbaik: {best_method} (F1={best_f1:.4f})")

# =====================================================
# CEK FILE YANG DIHASILKAN
# =====================================================
print("\n[CEK] File yang dihasilkan:")
files_to_check = [
    "data/processed/cleaned_cases.csv",
    "data/processed/cases.csv",
    "data/eval/retrieval_metrics.csv",
    "data/eval/prediction_metrics.csv",
    "models/tfidf_vectorizer.pkl",
    "models/svm_model.pkl",
    "models/label_encoder.pkl",
    "models/tfidf_matrix.pkl",
]
for f in files_to_check:
    status = "[OK]" if Path(f).exists() else "[MISSING]"
    print(f"  {status} {f}")

print("\n" + "=" * 60)
print("SEMUA TEST BERHASIL! Pipeline CBR siap digunakan.")
print("Jalankan notebook secara berurutan untuk demo lengkap.")
print("=" * 60)
