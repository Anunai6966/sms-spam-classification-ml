# 🛡️ SMS Spam Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat)

## 🚀 Live Demo
**👉 [Try the app here](https://anunai-sms-spam-classification-ml.streamlit.app/)**

Type any SMS message and the model will instantly tell you if it's **Spam** or **Ham** with a confidence score.

---

## 📌 Project Overview

An end-to-end machine learning project that classifies SMS messages as **Spam** or **Ham** (not spam). Built with a complete production-ready pipeline — from raw data to a deployed web application.

This project goes beyond a Jupyter notebook — it includes modular Python code, a saved model, and a live deployed app.

---

## 📂 Project Structure

```
sms-spam-classification-ml/
├── data/
│   └── spam.csv                  # SMS Spam Collection Dataset
├── models/
│   ├── spam_model.pkl            # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
├── notebooks/
│   └── sms_spam_classification.ipynb  # Exploratory analysis
├── preprocessing.py              # Text cleaning functions
├── train.py                      # Model training + saving
├── predict.py                    # Prediction on new messages
├── evaluate.py                   # Model evaluation metrics
├── app.py                        # Streamlit web application
└── requirements.txt              # Project dependencies
```

---

## 🧠 ML Pipeline

```
Raw SMS Data
     ↓
Text Preprocessing (lowercase, remove punctuation, stopword removal)
     ↓
Feature Extraction (TF-IDF Vectorizer, 3000 features)
     ↓
Train/Test Split (80/20, stratified)
     ↓
Model Training (Logistic Regression vs Naive Bayes)
     ↓
Evaluation (Precision, Recall, F1-Score, Confusion Matrix)
     ↓
Save Model (pickle)
     ↓
Deploy (Streamlit Cloud)
```

---

## 📊 Results

| Model | Accuracy | Spam Precision | Spam Recall | F1-Score |
|-------|----------|---------------|-------------|----------|
| Naive Bayes | 97% | 0.99 | 0.80 | 0.88 |
| **Logistic Regression** | **97%** | **0.98** | **0.81** | **0.89** |

**Final model: Logistic Regression** — chosen for better recall and F1-score on the imbalanced dataset.

> ⚠️ Note: Accuracy alone is misleading here — the dataset has 87% ham and 13% spam (class imbalance). F1-score is the more reliable metric.

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML | Scikit-learn |
| NLP | NLTK, TF-IDF |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 🛠️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Anunai6966/sms-spam-classification-ml.git
cd sms-spam-classification-ml
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train.py
```

**4. Run the app**
```bash
python -m streamlit run app.py
```

---

## 🔍 Key Learnings

- Why **F1-score matters more than accuracy** on imbalanced datasets
- How to build a **modular ML pipeline** with reusable Python files
- Importance of saving **both the model AND the vectorizer** with pickle
- How to **deploy a machine learning model** as a live web application

---

## 🚀 Future Improvements

- [ ] Improve spam recall by tuning classification threshold
- [ ] Add a FastAPI backend for REST API access
- [ ] Experiment with advanced models (SVM, BERT)
- [ ] Add model monitoring to track performance over time
- [ ] Containerize with Docker

---

## 👨‍💻 Author

**Anunai Sai Goud**
- GitHub: [@Anunai6966](https://github.com/Anunai6966)
- Email: anunai6966l@gmail.com