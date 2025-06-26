import pandas as pd
import re
import collections
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ======================= Kelas SpellChecker =========================
# Digunakan untuk memperbaiki kata yang tidak baku dalam teks
class spellCheck:
    def train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def __init__(self):
        with open('id_dict/spellingset.txt', encoding='utf-8') as f:
            self.NWORDS = self.train(self.words(f.read()))
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def words(self, text):
        return re.findall('[a-z]+', text.lower())

    def edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts = [a + c + b for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words):
        return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        return max(candidates, key=self.NWORDS.get)


# ======================= Kelas SentiStrength =========================
# Untuk analisis sentimen berbasis leksikon dan aturan
class sentiStrength:
    def __init__(self):
        # Memuat kamus sentimen, emotikon, penguat, penyangkal, dsb.
        self.__sentiDict = self.load_dict('id_dict/sentimentword.txt', delimiter="\t")
        self.__emotDict = self.load_dict('id_dict/emoticon.txt', delimiter="\t")
        self.__negatingDict = self.load_list('id_dict/negatingword.txt')
        self.__boosterDict = self.load_dict('id_dict/boosterword.txt')
        self.__idiomDict = self.load_dict('id_dict/idiom.txt', delimiter="\t")
        self.__questionDict = self.load_list('id_dict/questionword.txt')
        self.__katadasar = self.load_list('id_dict/rootword.txt')

        # Memuat daftar kata kasar
        with open('KamusKataKasar.txt', encoding='utf-8') as f:
            self.__kamusKasar = set(k.strip().lower() for k in f if k.strip())

        # Inisialisasi penghitung sentimen
        self.p = 0  # positif
        self.n = 0  # negatif
        self.nn = 0  # netral

    def load_list(self, path):
        with open(path, encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def load_dict(self, path, delimiter="\t"):
        scores = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) == 2:
                    term, score = parts
                    if term not in scores:
                        try:
                            scores[term] = int(score)
                        except ValueError:
                            print(f"⚠️ Format tidak valid: {line.strip()}")
        return scores

    def main(self, text):
        # Menghitung skor sentimen berdasarkan kata, emotikon, dan idiom
        sentimen = self.__sentiDict
        emoticon = self.__emotDict
        booster = self.__boosterDict
        idiom = self.__idiomDict

        pos = 1  # nilai maksimal positif
        neg = -1  # nilai maksimal negatif
        isQuestion = False
        sentence_score = []

        tweet = text.split()

        # Hitung jumlah kata kasar dalam tweet
        jumlah_kata_kasar = sum(1 for w in tweet if w in self.__kamusKasar)

        for term_index, term in enumerate(tweet):
            score = 0
            bigram = f"{tweet[term_index-1]} {term}" if term_index > 0 else ""
            term = re.sub(r'(\w+)-(\w+)', r'\1', term)

            if term.isalpha():
                if term in sentimen:
                    score = sentimen[term]
                    # Pengecekan penguat dan penyangkal
                    if term_index > 0 and tweet[term_index - 1] in self.__negatingDict:
                        score = -abs(score) if score > 0 else abs(score)
                    if term_index > 0 and tweet[term_index - 1] in booster:
                        score += booster[tweet[term_index - 1]]
                    elif term_index < len(tweet) - 1 and tweet[term_index + 1] in booster:
                        score += booster[tweet[term_index + 1]]
                    if term_index > 0 and tweet[term_index - 1] in sentimen:
                        if score >= 3:
                            score += 1
                        elif score <= -3:
                            score -= 1
                if bigram in idiom:
                    score = idiom[bigram]
                if term in self.__questionDict:
                    isQuestion = True
            else:
                if term in emoticon:
                    score = emoticon[term]
                elif '?' in term:
                    isQuestion = True
                elif '!' in term:
                    score = 2
                elif re.sub(r'(\w+)[!]+$', r'\1', term) in sentimen:
                    base = re.sub(r'(\w+)[!]+$', r'\1', term)
                    score = sentimen[base]
                    score += 1 if score > 0 else -1

            pos = score if score > pos else pos
            neg = score if score < neg else neg
            if score != 0:
                term = f"{term} [{score}]"
            sentence_score.append(term)

        # Tentukan label sentimen
        if abs(pos) > abs(neg):
            self.countSentimen("+")
            senti_label = "positif"
        elif abs(pos) < abs(neg) and not isQuestion:
            self.countSentimen("-")
            senti_label = "negatif"
        else:
            self.countSentimen("?")
            senti_label = "netral"

        # Deteksi cyberbullying
        if senti_label == "negatif" and (neg <= -3 or jumlah_kata_kasar > 0):
            cyberbullying_label = "Cyberbullying"
        else:
            cyberbullying_label = "Non-Cyberbullying"

        # Penentuan tingkat keparahan
        if cyberbullying_label == "Cyberbullying":
            if jumlah_kata_kasar >= 3 or neg <= -4:
                tingkat_keparahan = "berat"
            elif jumlah_kata_kasar == 2 or neg == -3:
                tingkat_keparahan = "sedang"
            elif jumlah_kata_kasar == 1 or neg in [-2, -1]:
                tingkat_keparahan = "ringan"
            else:
                tingkat_keparahan = "ringan"
        else:
            tingkat_keparahan = "tidak ada"

        # Kembalikan hasil analisis dalam bentuk dictionary
        return {
            "teks_dengan_skor": ' '.join(sentence_score),
            "skor_pos": pos,
            "skor_neg": neg,
            "sentimen": senti_label,
            "labeling": cyberbullying_label,
            "keparahan": tingkat_keparahan,
            "jumlah_kata_kasar": jumlah_kata_kasar
        }

    def countSentimen(self, res):
        if res == "+":
            self.p += 1
        elif res == "-":
            self.n += 1
        else:
            self.nn += 1

    def getSentimenScore(self):
        return f"[positif:{self.p}] [negatif:{self.n}] [netral:{self.nn}]"


# ======================= Fungsi Utama Program =========================
def main():
    # Baca dataset hasil preprocessing
    df = pd.read_csv("preprocessing.csv")
    ss = sentiStrength()
    hasil_list = []

    # Lakukan analisis per baris
    for _, row in df.iterrows():
        try:
            token_list = ast.literal_eval(row['tweet_bersih'])
            text = ' '.join(token_list)
        except:
            text = str(row['tweet_bersih'])

        hasil = ss.main(text)

        hasil_list.append({
            "text": text,
            "skor_pos": hasil["skor_pos"],
            "skor_neg": hasil["skor_neg"],
            "sentimen": hasil["sentimen"],
            "labeling": hasil["labeling"],
            "keparahan": hasil["keparahan"],
            "jumlah_kata_kasar": hasil["jumlah_kata_kasar"],
            "teks_dengan_skor": hasil["teks_dengan_skor"]
        })

    # Simpan hasil ke file CSV
    hasil_df = pd.DataFrame(hasil_list)
    hasil_df.to_csv("sentistrength.csv", index=False)
    print("✅ Analisis selesai. File disimpan: sentistrength.csv")
    print(ss.getSentimenScore())

# ======================= Visualisasi =========================
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 4))

# Grafik 1: Distribusi Sentimen
    plt.subplot(1, 3, 1)
    ax1 = sns.countplot(data=hasil_df, x="sentimen", palette="coolwarm")
    plt.title("Distribusi Sentimen")
    plt.xlabel("Kategori Sentimen")
    plt.ylabel("Jumlah Tweet")
    for p in ax1.patches:
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width() / 2., height + 2, int(height), ha="center", fontsize=9)

# Grafik 2: Deteksi Cyberbullying
    plt.subplot(1, 3, 2)
    ax2 = sns.countplot(data=hasil_df, x="labeling", palette="Set2")
    plt.title("Deteksi Cyberbullying")
    plt.xlabel("Label")
    plt.ylabel("Jumlah Tweet")
    for p in ax2.patches:
        height = p.get_height()
        ax2.text(p.get_x() + p.get_width() / 2., height + 2, int(height), ha="center", fontsize=9)

# Grafik 3: Tingkat Keparahan
    plt.subplot(1, 3, 3)
    ax3 = sns.countplot(data=hasil_df, x="keparahan", 
                    order=["tidak ada", "ringan", "sedang", "berat"], palette="magma")
    plt.title("Tingkat Keparahan Cyberbullying")
    plt.xlabel("Kategori")
    plt.ylabel("Jumlah Tweet")
    for p in ax3.patches:
        height = p.get_height()
        ax3.text(p.get_x() + p.get_width() / 2., height + 2, int(height), ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()

# Jalankan program
if __name__ == "__main__":
    main()
