# Modul 1 - Text Preprocessing

## 📌 Overview

Modul ini dirancang untuk membersihkan dan menyiapkan teks agar dapat diproses secara optimal dalam **Natural Language Processing (NLP)**. Preprocessing yang tepat meningkatkan akurasi dan efisiensi model, terutama dalam tugas seperti **Named Entity Recognition (NER), klasifikasi teks, dan analisis sentimen**.

## 🔄 Tahapan Preprocessing

### 1⃣ Tokenisasi

**🛠 Tujuan:** Memecah teks menjadi unit yang lebih kecil seperti kata atau kalimat untuk mempermudah pemrosesan NLP.

#### ✅ Kelebihan:
- Menyusun teks agar lebih mudah diproses.
- Membantu ekstraksi fitur untuk model NLP.

#### ❌ Kekurangan:
- Kehilangan konteks jika pemisahan tidak akurat.
- Kata majemuk kompleks memerlukan tokenisasi yang lebih canggih.

#### ⚖️ Perbandingan Pustaka:
- **NLTK:** Menyediakan berbagai metode tokenisasi berbasis aturan.
- **SpaCy:** Lebih cepat dan akurat karena menggunakan model machine learning.

#### 💻 Implementasi:

**Input:**

```python
text = "Python programming is very popular among developers."
```

**Menggunakan NLTK:**

```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens)  # Output: ['Python', 'programming', 'is', 'very', 'popular', 'among', 'developers', '.']
```

**Menggunakan SpaCy:**

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
tokens = [token.text for token in doc]
print(tokens)  # Output: ['Python', 'programming', 'is', 'very', 'popular', 'among', 'developers', '.']
```

---

### 2⃣ Casefolding

**🛠 Tujuan:** Mengubah teks menjadi huruf kecil untuk menghindari perbedaan makna akibat kapitalisasi.

#### ✅ Manfaat:
- Standarisasi teks untuk konsistensi.
- Menghindari kesalahan akibat perbedaan kapitalisasi.

#### 💻 Implementasi:

**Input:**

```python
text = "Python Is Useful In Data Science."
```

**Menggunakan Python Standard:**

```python
text_lower = text.lower()
print(text_lower)  # Output: 'python is useful in data science.'
```

---

### 3⃣ Stemming & Lemmatization

**🛠 Tujuan:** Mengubah kata ke bentuk dasarnya dengan menghapus akhiran (stemming) atau mempertimbangkan aturan linguistik (lemmatization).

#### ✅ Mengapa dilakukan?
1. **Digunakan dalam Information Retrieval (IR)** untuk memperluas pencarian:
   - *organize, organizes, organizing* → *organize*
   - *democracy, democratic, democratization* → *democracy*
2. **Mengurangi dimensi dalam Text Mining**, meningkatkan akurasi dalam beberapa kasus.
3. **Dalam IR, meningkatkan recall tapi menurunkan akurasi**, contoh:
   - *operate, operating, operates, operation* → *operate* memperluas pencarian tetapi mengurangi spesifikasi hasil.

#### 💻 Implementasi Stemming:

**Input:**

```python
text = "playing running jumping"
```

**Menggunakan NLTK:**

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in text.split()]
print(stems)  # Output: ['play', 'run', 'jump']
```

#### 💻 Implementasi Lemmatization:

**Input:**

```python
text = "running jumped playing"
```

**Menggunakan NLTK:**

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word, pos="v") for word in text.split()]
print(lemmas)  # Output: ['run', 'jump', 'play']
```

---

### 4⃣ Stopword Removal

**🛠 Tujuan:** Menghapus kata-kata umum yang tidak memiliki makna signifikan, seperti "dan", "di", "ke".

#### ✅ Manfaat:
- Meningkatkan efisiensi model NLP dengan menyimpan kata yang lebih relevan.
- Mengurangi dimensi data untuk model yang lebih ringan.

#### 💻 Implementasi:

**Input:**

```python
text = "Saya ingin belajar NLP dan memahami stopwords."
```

**Menggunakan NLTK:**

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words("indonesian"))
tokens = [word for word in text.split() if word.lower() not in stop_words]
print(tokens)  # Output: ['ingin', 'belajar', 'NLP', 'memahami', 'stopwords.']
```

---

### 5⃣ Word Cloud

**🛠 Tujuan:** Menampilkan visualisasi frekuensi kata dalam teks.

#### ✅ Manfaat:
- Memudahkan identifikasi kata yang sering muncul.
- Membantu analisis tren kata kunci dalam teks besar.

#### 💻 Implementasi:

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

**Contoh Output Word Cloud:**
![alt text](/assets/image.png)
---

## 📚 Informasi Tambahan

### 🔹 Pustaka NLP yang Digunakan:

| Pustaka | Penggunaan Utama |
|----------|-----------------|
| **NLTK**  | Tokenisasi, stemming, lemmatization dasar |
| **SpaCy** |  Named Entity Recognition (NER), dependency parsing |
| **TextBlob** | Analisis sentimen, koreksi ejaan |
| **Sastrawi** | Stemming Bahasa Indonesia dan Stopwords Bahas Indonesia|
| **Gensim**  | Word embeddings, LDA topic modeling |


## 📝 Tugas 💡

🔹 **Melakukan preprocessing text pada dataset pada [Kaggle](https://www.kaggle.com/).**

🔹 **Menambahkan minimal 3 teknik preprocessing tambahan** yang belum disebutkan di atas.

🚀 **Silakan implementasikan dan dokumentasikan hasilnya dalam bentuk github repository!**


## Referensi : 
1. [Complete Guide to Text Preprocessing in NLP](https://medium.com/@devangchavan0204/complete-guide-to-text-preprocessing-in-nlp-b4092c104d3e)
2. [Mastering Stemming Algorithms in Natural Language Processing: A Complete Guide with Python Implementation](https://medium.com/@omrylmzz35/mastering-stemming-algorithms-in-natural-language-processing-a-complete-guide-with-python-e7fd12089a69)
3. [Getting started with Text Preprocessing](https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing)



