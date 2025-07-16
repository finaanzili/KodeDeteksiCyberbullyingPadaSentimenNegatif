import pandas as pd
import re
import collections
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ======================= Kelas SpellChecker =========================
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
class sentiStrength:
    def __init__(self):
        self.__sentiDict = self.load_dict('id_dict/sentimentword.txt', delimiter="\t")
        self.__emotDict = self.load_dict('id_dict/emoticon.txt', delimiter="\t")
        self.__negatingDict = self.load_list('id_dict/negatingword.txt')
        self.__boosterDict = self.load_dict('id_dict/boosterword.txt')
        self.__idiomDict = self.load_dict('id_dict/idiom.txt', delimiter="\t")
        self.__questionDict = self.load_list('id_dict/questionword.txt')
        self.__katadasar = self.load_list('id_dict/rootword.txt')

        with open('KamusKataKasar.txt', encoding='utf-8') as f:
            self.__kamusKasar = set(k.strip().lower() for k in f if k.strip())

        self.p = 0
        self.n = 0
        self.nn = 0

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
        sentimen = self.__sentiDict
        emoticon = self.__emotDict
        booster = self.__boosterDict
        idiom = self.__idiomDict

        pos = 1
        neg = -1
        isQuestion = False
        sentence_score = []
        tweet = text.split()

        jumlah_kata_kasar = sum(1 for w in tweet if w in self.__kamusKasar)

        for term_index, term in enumerate(tweet):
            score = 0
            bigram = f"{tweet[term_index-1]} {term}" if term_index > 0 else ""
            term = re.sub(r'(\w+)-(\w+)', r'\1', term)

            if term.isalpha():
                if term in sentimen:
                    score = sentimen[term]
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

        if abs(pos) > abs(neg):
            self.countSentimen("+")
            senti_label = "positif"
        elif abs(pos) < abs(neg) and not isQuestion:
            self.countSentimen("-")
            senti_label = "negatif"
        else:
            self.countSentimen("?")
            senti_label = "netral"

        # Langsung klasifikasi untuk sentimen negatif
        if senti_label == "negatif":
            cyberbullying_label = "Cyberbullying"
            if jumlah_kata_kasar >= 3 or neg <= -4:
                tingkat_keparahan = "berat"
            elif jumlah_kata_kasar == 2 or neg == -3:
                tingkat_keparahan = "menengah"
            elif jumlah_kata_kasar == 1 or neg in [-2, -1]:
                tingkat_keparahan = "ringan"
            else:
                tingkat_keparahan = "ringan"
        else:
            cyberbullying_label = "Non-Cyberbullying"
            tingkat_keparahan = "tidak ada"

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
    df = pd.read_csv("preprocessingfiks_3.csv")
    ss = sentiStrength()
    hasil_list = []

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

    hasil_df = pd.DataFrame(hasil_list)
    hasil_df.to_csv("sentistrengthfiks_3.csv", index=False)
    print("✅ Analisis selesai. File disimpan: sentistrengthfiks_3.csv")
    print(ss.getSentimenScore())

    # ======================= Visualisasi =========================
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))

    # Grafik 1: Distribusi Sentimen
    plt.subplot(1, 2, 1)
    ax1 = sns.countplot(data=hasil_df, x="sentimen", palette="coolwarm")
    plt.title("Distribusi Sentimen")
    plt.xlabel("Kategori Sentimen")
    plt.ylabel("Jumlah Tweet")
    for p in ax1.patches:
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width() / 2., height + 2, int(height), ha="center", fontsize=9)

    # Grafik 2: Distribusi Tingkat Keparahan
    plt.subplot(1, 2, 2)
    hasil_keparahan = hasil_df[hasil_df["keparahan"] != "tidak ada"]
    ax2 = sns.countplot(data=hasil_keparahan, x="keparahan", 
                        order=["ringan", "menengah", "berat"], palette="magma")
    plt.title("Tingkat Keparahan Cyberbullying")
    plt.xlabel("Kategori Keparahan")
    plt.ylabel("Jumlah Tweet")
    for p in ax2.patches:
        height = p.get_height()
        ax2.text(p.get_x() + p.get_width() / 2., height + 1, int(height), ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()

# Jalankan program
if __name__ == "__main__":
    main()
