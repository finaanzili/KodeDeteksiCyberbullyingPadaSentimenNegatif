import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

# ----------------- 1. Load Data Preprocessing -----------------
df_pre = pd.read_csv('preprocessingfiks_3.csv')  # hanya file preprocessing (tweet_bersih)

# ----------------- 2. Ubah tweet_bersih (string list) menjadi list token -----------------
df_pre['tweet_tokens'] = df_pre['tweet_bersih'].apply(ast.literal_eval)

# ----------------- 3. TF-IDF Vectorizer -----------------
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,  # wajib None jika pakai tokenizer custom
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    ngram_range=(1, 2)
)

# Fit dan transform TF-IDF
tfidf_matrix = vectorizer.fit_transform(df_pre['tweet_tokens'])

# ----------------- 4. Buat DataFrame TF-IDF -----------------
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# ----------------- 5. Tambahkan Kolom Asli -----------------
tfidf_df['tweet_tokens'] = df_pre['tweet_tokens'].apply(lambda x: ' '.join(x))

# ----------------- 6. Hitung TF-IDF rata-rata -----------------
tfidf_only_cols = [col for col in tfidf_df.columns if col != 'tweet_tokens']
tfidf_df['tfidf'] = tfidf_df[tfidf_only_cols].mean(axis=1)

# ----------------- 7. Simpan Hasil -----------------
tfidf_df.to_csv('tfidf_fiks_3.csv', index=False)
print("✅ TF-IDF tanpa label berhasil dibuat dan disimpan di 'tfidf_fiks_3.csv'")
