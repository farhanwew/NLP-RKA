# Teknik Pemrosesan Teks

Modul ini menyediakan pengenalan praktis tentang teknik utama pemrosesan teks yang digunakan dalam Pemrosesan Bahasa Alami (NLP). Teknik-teknik berikut dibahas:

### Persyaratan
Pastikan Anda telah menginstal pustaka yang diperlukan sebelum menjalankan notebook:

```bash
pip install numpy pandas nltk scikit-learn gensim matplotlib
```

## 1. Bag of Words (BoW)
**Tujuan:** Mengubah teks menjadi representasi numerik dengan menghitung frekuensi kemunculan setiap kata dalam dokumen tanpa memperhatikan urutan kata.

Diimplementasikan menggunakan `CountVectorizer` dari `sklearn`.

### Contoh Kode:
```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

corpus = [
    "Natural Language Processing is amazing!",
    "Deep Learning makes NLP more powerful.",
    "Word embeddings capture semantic meaning.",
    "TF-IDF is a great technique for text representation."
]

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(corpus)
print(pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out()))
```

## 2. TF-IDF (Term Frequency - Inverse Document Frequency)
**Tujuan:** Memberikan bobot pada kata berdasarkan frekuensinya dalam dokumen relatif terhadap keseluruhan korpus, sehingga kata yang umum muncul di banyak dokumen memiliki bobot lebih rendah.

Diimplementasikan menggunakan `TfidfVectorizer`.

### Contoh Kode:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(corpus)
print(pd.DataFrame(X_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out()))
```

## 3. Distribusi Frekuensi Kata
**Tujuan:** Menghitung dan menganalisis seberapa sering kata-kata tertentu muncul dalam teks untuk memahami pola kata yang dominan.

Menggunakan `FreqDist` dari NLTK.

### Contoh Kode:
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

nltk.download('punkt')

all_words = word_tokenize(" ".join(corpus).lower())
freq_dist = FreqDist(all_words)
print(freq_dist.most_common(10))

plt.figure(figsize=(10, 4))
freq_dist.plot(10, cumulative=False)
plt.show()
```

## 4. N-grams
**Tujuan:** Menangkap hubungan kata berdasarkan urutan kata dalam teks dengan membuat sekelompok kata yang berurutan (bigrams, trigrams, dll.).

### Contoh Kode:
```python
from nltk.util import ngrams

def generate_ngrams(text, n):
    words = word_tokenize(text.lower())
    return list(ngrams(words, n))

bigrams = generate_ngrams(" ".join(corpus), 2)
trigrams = generate_ngrams(" ".join(corpus), 3)

print("Contoh Bigrams:", bigrams[:5])
print("Contoh Trigrams:", trigrams[:5])
```

## 5. Word Embeddings (Word2Vec)
**Tujuan:** Mewakili kata dalam bentuk vektor numerik yang mempertahankan hubungan semantik antar kata, memungkinkan pemahaman konteks kata dalam teks.

Menggunakan `Word2Vec` dari `gensim`.

### Contoh Kode:
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Pastikan corpus dalam bentuk list
tokenized_corpus = corpus.astype(str).apply(lambda x: word_tokenize(x.lower())).tolist()

# Melatih model Word2Vec
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=10, window=2, min_count=1, workers=4)

# Cek vektor untuk kata tertentu
word = "fantastic"
if word in word2vec_model.wv:
    print(f"Vektor Word2Vec untuk '{word}':\n", word2vec_model.wv[word])
else:
    print(f"Kata '{word}' tidak ditemukan dalam model.")
```
### Penjelasan Parameter Word2Vec:
- **`sentences=tokenized_corpus`**: Daftar kalimat yang telah di-tokenisasi (setiap kalimat direpresentasikan sebagai daftar kata). Model belajar dari token-token ini untuk membangun representasi kata dalam bentuk vektor.
- **`vector_size=10`**: Menentukan dimensi vektor setiap kata. Semakin besar nilainya, semakin kaya representasi semantik, tetapi juga lebih berat dalam komputasi.
- **`window=2`**: Menentukan konteks kata dalam satu kalimat (jumlah kata sebelum dan sesudah kata target yang diperhitungkan dalam pembelajaran). Dengan `window=2`, model hanya melihat dua kata sebelum dan dua kata setelah kata target dalam satu kalimat.
- **`min_count=1`**: Kata-kata yang muncul kurang dari 1 kali akan diabaikan (dalam hal ini semua kata tetap dipertahankan). Jika `min_count=5`, kata yang muncul kurang dari 5 kali akan diabaikan.
- **`workers=4`**: Menentukan jumlah thread CPU yang digunakan untuk mempercepat proses training. Disarankan untuk menggunakan nilai yang sesuai dengan jumlah core CPU yang tersedia.



### Output
- **Bag of Words & TF-IDF**: Representasi teks dalam bentuk tabel.
- **Distribusi Frekuensi**: Statistik kata yang umum dan grafik distribusi frekuensi.
- **N-grams**: Contoh bigrams dan trigrams.
- **Word Embeddings**: Representasi kata dalam bentuk vektor.

Repositori ini berfungsi sebagai panduan dasar untuk pemrosesan teks dalam proyek NLP. 🚀



