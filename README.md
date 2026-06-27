# 📩 Spam Email Detection System

A simple web app that detects whether an email/message is **Spam** or **Not Spam** using Machine Learning. Built with **Streamlit** and **Naive Bayes (TF-IDF)**.

## 🚀 Features

- Detects spam vs non-spam messages in real time
- Shows spam probability and confidence score
- Displays dataset overview (total emails, spam count, model accuracy)
- Simple and clean web interface

## 🛠️ Tech Stack

- Python
- Streamlit
- scikit-learn (TF-IDF + Multinomial Naive Bayes)
- pandas

## 📂 Project Structure

```
.
├── spam_app.py     # Main Streamlit app
├── spam.csv        # Dataset (SMS Spam Collection)
└── README.md
```

## ⚙️ Installation & Setup

1. Clone the repository

```bash
git clone https://github.com/akshayraja20/Spam-email-detection.git
cd Spam-email-detection
```

2. Install the required libraries

```bash
pip install streamlit pandas scikit-learn
```

3. Run the app

```bash
streamlit run spam_app.py
```

4. The app will open automatically in your browser at `http://localhost:8501`

## 📊 How It Works

1. The dataset (`spam.csv`) is loaded and labeled as `ham` (0) or `spam` (1)
2. Text is converted into numerical features using **TF-IDF Vectorization**
3. A **Multinomial Naive Bayes** model is trained on this data
4. User can enter any email/message text, and the model predicts whether it's spam along with a confidence score

## 📌 Note

This project is for learning/educational purposes and demonstrates a basic spam classification pipeline.
