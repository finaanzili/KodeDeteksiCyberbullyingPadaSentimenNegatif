import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# === Load dataset tweet ===
data = pd.read_csv('DataCyber2425.csv')  # Pastikan kolom 'full_text' tersedia

# === Inisialisasi stemmer dari Sastrawi ===
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# === Load kamus normalisasi ===
kamus_norm = pd.read_csv('kamus_normalisasi.csv')  # Kolom: slang, formal
normalisasi_dict = dict(zip(kamus_norm['slang'], kamus_norm['formal']))

# === Load kamus kata kasar dan lakukan stemming ===
with open('KamusKataKasar.txt', 'r', encoding='utf-8') as f:
    daftar_kasar = {stemmer.stem(k.strip().lower()) for k in f if k.strip()}

# === Ambil stopwords bahasa Indonesia dari NLTK ===
stopword_list = set(stopwords.words('indonesian'))

# === Lindungi kata negasi dari penghapusan stopword ===
kata_negasi = {'tidak', 'jangan', 'ga', 'gak', 'engga', 'enggak', 'nggak', 'ngga', 'bukan', 'tak'}
stopword_list = stopword_list - kata_negasi  # Stopword final tanpa kata negasi

# === Fungsi preprocessing lengkap bertahap ===
def preprocessing_bertahap(text):
    hasil = {}
    
    # 1. Case folding (ubah huruf besar ke kecil)
    text = text.lower()
    hasil['casefolding'] = text

    # 2. Cleansing (hapus URL, mention, hashtag)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r'#\w+', '', text)
    hasil['cleansing'] = text

    # 3. Tokenisasi awal (sebelum normalisasi)
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        print(f"Tokenisasi gagal: {text}\nError: {e}")
        tokens = []
    hasil['tokenizing_awal'] = tokens

    # 4. Normalisasi slang ke bentuk formal
    tokens = [normalisasi_dict.get(w, w) for w in tokens]
    hasil['normalisasi'] = tokens

    # ✅ 5. Tokenisasi ulang setelah normalisasi (agar teks hasil normalisasi ditokenisasi ulang)
    text_norm = ' '.join(tokens)
    tokens = word_tokenize(text_norm)
    hasil['tokenizing_setelah_normalisasi'] = tokens

    # 6. Stopword removal (hapus kata umum, tetapi tetap simpan kata negasi)
    tokens = [w for w in tokens if w not in stopword_list and len(w) > 2]
    hasil['stopword_removal'] = tokens

    # 7. Stemming (ubah kata ke bentuk dasar)
    tokens = [stemmer.stem(w) for w in tokens]
    hasil['stemming'] = tokens

    # 8. Final tweet bersih
    hasil['tweet_bersih'] = tokens

    return pd.Series(hasil)

# === Terapkan preprocessing ke seluruh data ===
data = data.dropna(subset=['full_text'])  # Hindari nilai kosong
data = data.reset_index(drop=True)

hasil = data['full_text'].astype(str).apply(preprocessing_bertahap)
data = pd.concat([data, hasil], axis=1)

# === Hitung jumlah kata kasar yang terdeteksi ===
def hitung_kasar(tokens):
    return sum(1 for w in tokens if w in daftar_kasar)

data['jumlah_kata_kasar'] = data['tweet_bersih'].apply(hitung_kasar)

# (Opsional) Simpan kata-kata kasar yang terdeteksi untuk analisis
data['kata_kasar_terdeteksi'] = data['tweet_bersih'].apply(lambda tokens: [w for w in tokens if w in daftar_kasar])

# === Simpan hasil preprocessing ===
data.to_csv('preprocessingfiks_3.csv', index=False)
print("✅ Preprocessing selesai. Hasil disimpan di 'preprocessingfiks_3.csv'")
