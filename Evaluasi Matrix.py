import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# ----------------- 1. Load Data -----------------
data = pd.read_csv('hasil_fuzzy_fiks_3.csv')  # pastikan file berisi kolom 'cyberbullying' & 'label_cyberbullying'

# ----------------- 2. Ambil label aktual dan prediksi -----------------
y_true = data['labeling'].astype(str).fillna('unknown')           # label asli dari SentiStrength
y_pred = data['label_cyberbullying'].astype(str).fillna('unknown')     # hasil prediksi dari fuzzy

# ----------------- 3. Confusion Matrix 2x2 -----------------
labels = ['Cyberbullying', 'Non-Cyberbullying']
cm = confusion_matrix(y_true, y_pred, labels=labels)
print("Confusion Matrix (2x2):\n", cm)

# ----------------- 4. Classification Report -----------------
report = classification_report(y_true, y_pred, labels=labels)
print("\nClassification Report:\n", report)

# ----------------- 5. Evaluasi Global -----------------
accuracy = accuracy_score(y_true, y_pred) * 100
precision = precision_score(y_true, y_pred, pos_label='Cyberbullying') * 100
recall = recall_score(y_true, y_pred, pos_label='Cyberbullying') * 100
f1 = f1_score(y_true, y_pred, pos_label='Cyberbullying') * 100

# ----------------- 6. Tampilkan metrik evaluasi -----------------
print(f"\nAkurasi     : {accuracy:.2f}%")
print(f"Precision   : {precision:.2f}%")
print(f"Recall      : {recall:.2f}%")
print(f"F1 Score    : {f1:.2f}%")

# ----------------- 7. Visualisasi Confusion Matrix -----------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix 2x2 (Cyberbullying Detection)')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.tight_layout()
plt.show()

# ----------------- 8. Diagram Batang Perbandingan -----------------
plt.figure(figsize=(7, 5))
actual_counts = y_true.value_counts().sort_index()
predicted_counts = y_pred.value_counts().sort_index()

df_vis = pd.DataFrame({
    'Aktual': actual_counts,
    'Prediksi': predicted_counts
}).fillna(0)

df_vis.plot(kind='bar', figsize=(7, 5))
plt.title('Perbandingan Label Aktual vs Prediksi (Cyberbullying)')
plt.xlabel('Label')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ----------------- 9. Simpan ke File -----------------
with open("evaluasi_matrix_fiks_3.txt", "w", encoding="utf-8") as f:
    f.write("=== Evaluasi Fuzzy Mamdani Berdasarkan Label Cyberbullying ===\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write(f"Akurasi  : {accuracy:.2f}%\n")
    f.write(f"Precision: {precision:.2f}%\n")
    f.write(f"Recall   : {recall:.2f}%\n")
    f.write(f"F1 Score : {f1:.2f}%\n")

print("✅ Hasil evaluasi disimpan ke 'evaluasi_matrix.txt'")
