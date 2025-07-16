import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# TAHAP 1: LOAD DATA & KAMUS
# ============================
df = pd.read_csv("tfidf_fiks_3.csv")  # harus ada kolom 'tfidf' dan 'text' atau 'tweet_tokens'

if 'tweet_tokens' not in df.columns:
    df['tweet_tokens'] = df['text'].astype(str).str.lower().str.split()

with open("KamusKataKasar.txt", "r", encoding="utf-8") as file:
    kamus_kasar = set([baris.strip().lower() for baris in file])

# ============================
# TAHAP 2: NORMALISASI TF-IDF
# ============================
df['tfidf_norm'] = (df['tfidf'] - df['tfidf'].min()) / (df['tfidf'].max() - df['tfidf'].min())

def hitung_kata_kasar(teks):
    kata = str(teks).lower().split()
    return sum(1 for k in kata if k in kamus_kasar)

df['jumlah_kata_kasar'] = df['tweet_tokens'].apply(hitung_kata_kasar)

# ============================
# TAHAP 3: FUZZY LOGIC SETUP (Disesuaikan)
# ============================
tfidf = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'TF-IDF')
keparahan = ctrl.Consequent(np.arange(0, 101, 1), 'Keparahan')

# Fungsi keanggotaan input TF-IDF
tfidf['rendah'] = fuzz.trimf(tfidf.universe, [0.0, 0.0, 0.3])
tfidf['sedang'] = fuzz.trimf(tfidf.universe, [0.2, 0.5, 0.8])
tfidf['tinggi'] = fuzz.trimf(tfidf.universe, [0.7, 1.0, 1.0])

# Fungsi keanggotaan output keparahan
keparahan['ringan'] = fuzz.trimf(keparahan.universe, [0, 0, 30])
keparahan['menengah'] = fuzz.trimf(keparahan.universe, [20, 50, 80])
keparahan['berat'] = fuzz.trimf(keparahan.universe, [70, 100, 100])

# Aturan fuzzy 
rule1 = ctrl.Rule(tfidf['rendah'], keparahan['ringan'])
rule2 = ctrl.Rule(tfidf['sedang'], keparahan['menengah'])
rule3 = ctrl.Rule(tfidf['tinggi'], keparahan['berat'])

keparahan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
keparahan_simulasi = ctrl.ControlSystemSimulation(keparahan_ctrl)

# ============================
# TAHAP 4: INFERENSI & PELABELAN
# ============================
hasil = []

for i, row in df.iterrows():
    val = row['tfidf_norm']
    n_kasar = row['jumlah_kata_kasar']

    keparahan_simulasi.input['TF-IDF'] = val
    keparahan_simulasi.compute()
    nilai = keparahan_simulasi.output['Keparahan']

    # ATURAN PELABELAN SENTIMEN (berdasarkan fuzzy dan kata kasar saja)
    if n_kasar > 0:
        label_sentimen = 'negatif'
    elif nilai < 30:
        label_sentimen = 'positif'
    elif nilai < 50:
        label_sentimen = 'netral'
    else:
        label_sentimen = 'negatif'

    # ATURAN PELABELAN CYBERBULLYING
    if label_sentimen == 'negatif':
        label_cb = 'Cyberbullying'
    else:
        label_cb = 'Non-Cyberbullying'

    # ATURAN KLASIFIKASI TINGKAT KEPARAHAN HANYA UNTUK NEGATIF (tanpa jumlah kata kasar)
    if label_cb == 'Cyberbullying' and label_sentimen == 'negatif':
        if nilai <= 30:
            label_keparahan = 'ringan'
        elif nilai <= 50:
            label_keparahan = 'menengah'
        else:
            label_keparahan = 'berat'
    else:
        label_keparahan = 'tidak ada'

    hasil.append((round(nilai, 2), label_sentimen, label_cb, label_keparahan))

# ============================
# TAHAP 5: HASIL KE DATAFRAME
# ============================
df['keparahan_nilai'] = [x[0] for x in hasil]
df['label_sentimen'] = [x[1] for x in hasil]
df['label_cyberbullying'] = [x[2] for x in hasil]
df['keparahan_label'] = [x[3] for x in hasil]

# ============================
# TAHAP 6: GROUND TRUTH (SENTISTRENGTH)
# ============================
try:
    df_senti = pd.read_csv("sentistrengthfiks_3.csv")
    df['labeling'] = df_senti['labeling']
except Exception as e:
    print("⚠️ Gagal memuat sentistrengthfiks_3.csv:", e)
    df['labeling'] = 'UNKNOWN'

# ============================
# TAHAP 7: SIMPAN HASIL
# ============================
df.to_csv("hasil_fuzzy_fiks_3.csv", index=False)
print("✅ Hasil akhir disimpan di hasil_fuzzy.csv")

# ============================
# TAHAP 8: VISUALISASI KEANGGOTAAN
# ============================
tfidf.view()
keparahan.view()
plt.show()

# ============================
# TAHAP 9: VISUALISASI HASIL INFERENSI
# ============================
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df[df['keparahan_label'] != 'tidak ada'], x='keparahan_label',
                   order=['ringan', 'menengah', 'berat'],
                   palette='coolwarm')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 5,
            f'{int(height)}', ha="center", fontsize=11, weight='bold')

plt.title('Distribusi Label Keparahan Hasil Fuzzy Mamdani')
plt.xlabel('Label Keparahan')
plt.ylabel('Jumlah Tweet')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('visualisasi_keparahan_fuzzy13.png')
plt.show()

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='label_cyberbullying', palette='Set1')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 5,
            f'{int(height)}', ha="center", fontsize=11, weight='bold')

plt.title("Distribusi Label Cyberbullying")
plt.xlabel("Label Cyberbullying")
plt.ylabel("Jumlah Tweet")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("label_cyberbullying_fuzzy.png")
plt.show()

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='label_sentimen',
                   palette='Set2',
                   order=['positif', 'netral', 'negatif'])

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 5,
            f'{int(height)}', ha="center", fontsize=11, weight='bold')

plt.title("Distribusi Label Sentimen (Fuzzy Output)")
plt.xlabel("Label Sentimen")
plt.ylabel("Jumlah Tweet")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("label_sentimen_fuzzy.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df[df['label_cyberbullying'] == 'Cyberbullying'], 
                x='jumlah_kata_kasar', y='keparahan_nilai',
                hue='keparahan_label', palette='Set2')
plt.title('Scatterplot: Kata Kasar vs Nilai Keparahan (Cyberbullying)')
plt.xlabel('Jumlah Kata Kasar')
plt.ylabel('Nilai Keparahan (0-100)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('scatter_keparahan_vs_kasar7.png')
plt.show()

print("\n📊 Jumlah Data per Label Sentimen:")
print(df['label_sentimen'].value_counts())

print("\n📊 Jumlah Data per Label Cyberbullying:")
print(df['label_cyberbullying'].value_counts())

print("\n📊 Jumlah Data per Label Keparahan:")
print(df['keparahan_label'].value_counts())
