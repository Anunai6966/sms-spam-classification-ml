# SMS Spam Detection using Machine Learning

## 📌 Project Overview
This project implements a text classification system to classify SMS messages as **Spam** or **Ham** using machine learning techniques. The goal is to demonstrate the complete ML pipeline including data preprocessing, feature extraction, model training, and evaluation.

## 📂 Dataset
- SMS Spam Collection Dataset
- Labels: `spam`, `ham`
- ~5,500 SMS messages

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn

## 🧠 Approach
1. Loaded and explored the dataset
2. Cleaned and preprocessed text data
3. Converted text into numerical features using **TF-IDF**
4. Trained two models:
   - Naive Bayes (baseline)
   - Logistic Regression (final model)
5. Evaluated using precision, recall, F1-score
6. Visualized results using confusion matrix

## 📊 Results
- Logistic Regression achieved better recall and F1-score compared to Naive Bayes
- Accuracy alone was not considered due to class imbalance

## 📌 Key Learnings
- Importance of preprocessing text data
- Handling imbalanced datasets
- Comparing multiple models for better decision-making

## 🚀 Future Improvements
- Deploy model using Streamlit
- Experiment with advanced NLP techniques
