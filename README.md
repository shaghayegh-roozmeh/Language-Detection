# Language-Detection using Classic Machine Learning Models & BERT
Detect the language of text using machine learning and NLP techniques. Trained on 22,000 samples across 22 languages. Compares model performance on raw vs. preprocessed data. Includes both traditional ML models and a BERT-based deep learning model.
# 🌍 Language Detection using Machine Learning & BERT

This project aims to detect the language of a given text using traditional machine learning models as well as a deep learning approach with BERT. The dataset includes **22,000 samples** spanning **22 different languages**. The goal is to compare the performance of different models on both **preprocessed** and **raw text data**.

---

## 📁 Dataset

- 22 languages
- 1,000 samples per language
- Each sample contains a short text and its language label

---

## 🧹 Preprocessing

Text preprocessing was performed using `nltk` and `spaCy`:
- Lowercasing
- Tokenization
- Stop word removal
- Punctuation and number removal
- Lemmatization or stemming (based on language)

Both **preprocessed** and **raw** versions of the data were used for model training and evaluation to compare effectiveness.

---

## 🧠 Models Used

### ✅ Traditional Machine Learning Models
- Naive Bayes
- Logistic Regression
- Random Forest
- Linear SVM
- Gaussian SVM

### 🔥 Deep Learning Model
- BERT (from Hugging Face's `transformers` library)

---

## 📊 Evaluation

- **K-Fold Cross-Validation** (e.g., 5-fold)
- Accuracy for each fold and mean accuracy
- Confusion matrix visualizations

### 🧪 Key Findings
- **TF-IDF + Raw Data**: Up to **91% accuracy**
- **TF-IDF + Preprocessed Data**: Up to **86% accuracy**
- BERT achieved high performance and required no manual preprocessing

---

## 🛠️ Tools & Libraries

- `scikit-learn`
- `nltk`, `spaCy` – preprocessing
- `gensim` – Word2Vec
- `transformers` – BERT
- `pandas`, `numpy`
- `matplotlib`, `seaborn` – visualization

