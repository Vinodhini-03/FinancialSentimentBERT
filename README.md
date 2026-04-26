# 📈 Financial News Sentiment Predictor

> 🔗 **Live Demo:** [https://finsent-ai.streamlit.app](https://finsent-ai.streamlit.app)
>
> ⚠️ Note: App may take 1-2 minutes to wake up on first visit (free tier).

A deep learning system that classifies financial tweets and news headlines as **Bearish 📉**, **Bullish 📈**, or **Neutral ➡️** using RNN-based models and FinBERT transformer — with an AI-powered explanation engine using **LLaMA 3.3 70B**.

---

## Problem Statement

Financial markets move on information. Thousands of tweets and news headlines are posted every day — far more than any human team can process. This project builds an automated NLP pipeline that reads any financial headline and instantly classifies it as Bearish, Bullish, or Neutral.

---

## ✨ App Features

- **Single Headline Prediction** — Analyze any financial headline instantly with confidence scores
- **Multi-Tweet Batch** — Predict multiple headlines at once with individual result cards
- **File Upload** — Upload CSV/TXT files and get downloadable batch predictions
- **Model Comparison** — Side-by-side performance analysis of all 4 models with live comparison
- **AI Explanation** — LLaMA 3.3 70B explains WHY each prediction was made
- **Smart Override** — Rule-based layer catches implicit bearish/bullish signals that models miss
- **90% Accuracy** on implicit financial sentiment test cases

---

## 🏗️ System Architecture

```
Input Text
    ↓
Rule-Based Override Layer    ← Catches implicit financial signals
    ↓
Deep Learning Model          ← SimpleRNN / LSTM / GRU / FinBERT
    ↓
Sentiment Prediction         ← Bearish / Bullish / Neutral
    ↓
LLaMA 3.3 70B (Groq)        ← Explains the prediction in plain English
```

---

## 📁 Project Structure

```
FinancialSentimentBERT/
│
├── data/
│   ├── sent_train.csv          ← Training data (9,535 samples)
│   ├── sent_valid.csv          ← Validation data (2,388 samples)
│   ├── train_cleaned.csv       ← Preprocessed train data
│   └── valid_cleaned.csv       ← Preprocessed valid data
│
├── notebooks/
│   ├── 01_EDA.ipynb            ← Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  ← Text cleaning & tokenization
│   ├── 03_Models.ipynb         ← RNN, LSTM, GRU training
│   └── 04_BERT.ipynb           ← FinBERT fine-tuning
│
├── models/                     ← Hosted on HuggingFace Hub
│   ├── word2idx.pkl
│   ├── idx2word.pkl
│   ├── config.json
│   ├── LSTM_best.pt
│   ├── GRU_best.pt
│   ├── SimpleRNN_best.pt
│   └── FinBERT_best.pt
│
├── app/
│   └── streamlit_app.py        ← Streamlit dashboard
│
├── assets/                     ← All plots and charts
├── reports/                    ← PDF comparison report
├── requirements.txt
├── runtime.txt
└── README.md
```

> 📦 **Model files** are hosted on [HuggingFace Hub](https://huggingface.co/Vin003/FinancialSentimentBERT) due to GitHub's 100MB file size limit. The app downloads them automatically at startup.

---

## 📊 Dataset

- **Source:** Twitter Financial News Sentiment (Hugging Face)
- **Task:** 3-class sentiment classification
- **Labels:** LABEL_0 → Bearish | LABEL_1 → Bullish | LABEL_2 → Neutral
- **Train:** 9,535 samples | **Validation:** 2,388 samples
- **Class Distribution:** Neutral 64.7% | Bullish 20.2% | Bearish 15.1%
- **Class Imbalance:** Handled via weighted cross-entropy loss

---

## 🧠 Models

 ------------------------------------------------------------------------------
| Model           | Accuracy | Macro F1 | Bearish F1 | Bullish F1 | Neutral F1 |
|-----------------|----------|----------|------------|------------|------------|
| 🏆**FinBERT**   | 87.90%   | 0.8409   | 0.7816     | 0.8218     | 0.9191     |
|   **LSTM**      | 80.23%   | 0.7355   | 0.6417     | 0.6899     | 0.8749     |
|   **GRU**       | 79.56%   | 0.7339   | 0.6257     | 0.7139     | 0.8623     |
|   **SimpleRNN** | 43.80%   | 0.3336   | 0.2196     | 0.1828     | 0.5983     |
 ------------------------------------------------------------------------------

---

## 🔑 Key Findings

- **SimpleRNN failed** due to vanishing gradient — Macro F1 only 0.33
- **Learning rate was critical** — LSTM needed lr=3e-4, not 1e-3 (F1 jumped from 0.26 → 0.71)
- **LSTM ≈ GRU** — Only 0.002 Macro F1 difference between the two
- **FinBERT dominates** — Domain pre-training gives ~8% accuracy gain over LSTM
- **All 4 models fail on implicit sentiment** — Motivates the hybrid rule-based + LLM system

---

## 🚀 Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/Vinodhini-03/FinancialSentimentBERT.git
cd FinancialSentimentBERT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API key

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

## 📓 Run Notebooks

Run in order:
1. `01_EDA.ipynb`
2. `02_Preprocessing.ipynb`
3. `03_Models.ipynb`
4. `04_BERT.ipynb`

---

## 🎯 Sample Predictions

 ----------------------------------------------------------------------------------------------------
|                               Input                                  |          Prediction         |
|----------------------------------------------------------------------|-----------------------------|
| "Apple cuts iPhone production forecast amid weak demand"             | 🔴 Bearish                  |
| "Amazon beats Q4 earnings estimates with record revenue growth"      | 🟢 Bullish                  |
| "Federal Reserve holds interest rates steady"                        | 🔵 Neutral                  |
| "CFO stepping down to pursue other opportunities"                    | 🔴 Bearish (Smart Override) |
| "Free cash flow conversion exceeded 110% for third consecutive year" | 🟢 Bullish (Smart Override) |
 ----------------------------------------------------------------------------------------------------

---

## 🛠️ Tech Stack

 -------------------------------------------------
|     Category    |              Tools            |
|-----------------|-------------------------------|
| Language        | Python 3.11                   |
| Deep Learning   | PyTorch 2.11                  |
| Transformers    | HuggingFace Transformers 4.46 |
| LLM Explanation | Groq (LLaMA 3.3 70B)          |
| NLP             | NLTK                          |
| Dashboard       | Streamlit                     |
| Visualization   | Matplotlib, Seaborn           |
| Model Hosting   | HuggingFace Hub               |
| Deployment      | Streamlit Community Cloud     |
| Version Control | Git, GitHub                   |
| GPU Training    | Google Colab (Tesla T4)       |
 -------------------------------------------------

---

## 📬 Contact

- **GitHub:** [github.com/Vinodhini-03](https://github.com/Vinodhini-03)
- **Email:** vinodhinisugumar03@gmail.com
- **Live App:** [finsent-ai.streamlit.app](https://finsent-ai.streamlit.app)