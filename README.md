# spam_classification
📧 Spam Email Classifier – Beginner NLP Project

This project is a basic machine learning model that classifies text messages or emails as **Spam** or **Not Spam**, based on simple NLP features. It was built as part of my learning journey into **Natural Language Processing (NLP)** and text classification.

---

## 🧠 What It Does

The model uses handcrafted features extracted from each message to identify spammy patterns. These include:

- 📌 Count of **unusual words** (not in English vocabulary)
- 🔢 Presence of **phone numbers**
- 🔗 Presence of **links (URLs)**
- ❗ Number of **punctuation marks**
- 🔠 Number of **uppercase words**

These features are passed to a classifier (e.g., Logistic Regression or Naive Bayes) trained using `scikit-learn`.

---

## 📚 What I’m Learning

As a beginner exploring NLP, this project helped me learn:

- 🧹 How to **preprocess text data**
- 🧾 Using **NLTK** to identify meaningful words
- 🧠 Building a basic **feature extractor** for text
- ⚙️ Training and evaluating an ML model using **Scikit-learn**
- 💾 Saving models using `pickle` for future use
- Using HTML and CSS for frontend
---

## 🛠 Tech Stack

- Python
- Scikit-learn
- NLTK
- Pandas, NumPy
- Regular Expressions (`re`)
- HTML
- CSS
