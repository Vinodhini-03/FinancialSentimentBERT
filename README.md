#  Financial News Sentiment Predictor

A deep learning system that classifies financial tweets and news headlines as **Bearish**, **Bullish**, or **Neutral** using RNN-based models and FinBERT transformer.

---

##  Problem Statement

Financial markets move on information. Thousands of tweets and news headlines are posted every day — far more than any human team can process. This project builds an automated NLP pipeline that reads any financial headline and instantly classifies it as Bearish, Bullish, or Neutral.

---

##  Project Structure

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
│   └── 04_BERT.ipynb           ← FinBERT evaluation
│
├── models/
│   ├── word2idx.pkl            ← Vocabulary mapping
│   ├── idx2word.pkl            ← Reverse vocabulary
│   ├── config.json             ← Model configuration
│   ├── LSTM_best.pt            ← Best LSTM checkpoint
│   ├── GRU_best.pt             ← Best GRU checkpoint
│   └── SimpleRNN_best.pt       ← Best SimpleRNN checkpoint
│   (FinBERT_best.pt excluded — 417MB, trained on Google Colab T4 GPU)
│
├── app/
│   └── streamlit_app.py        ← Streamlit dashboard
│
├── assets/                     ← All plots and charts
├── reports/                    ← PDF comparison report
├── requirements.txt
└── README.md

---

##  Dataset

- **Source:** Twitter Financial News Sentiment (Hugging Face)
- **Task:** 3-class sentiment classification
- **Labels:** LABEL_0 → Bearish | LABEL_1 → Bullish | LABEL_2 → Neutral
- **Train:** 9,535 samples | **Validation:** 2,388 samples
- **Class distribution:** Neutral 64.7% | Bullish 20.2% | Bearish 15.1%

---

##  Models

 ------------------------------------------------------------------------------
|    Model    |  Accuracy  |  Macro F1  | Bearish F1 | Bullish F1 | Neutral F1 |
|-------------|------------|------------|------------|------------|------------|
|  SimpleRNN  |   0.4380   |   0.3336   |   0.2196   |   0.1828   |   0.5983   |
|     GRU     |   0.7956   |   0.7339   |   0.6257   |   0.7139   |   0.8623   |
|    LSTM     |   0.8023   |   0.7355   |   0.6417   |   0.6899   |   0.8749   |
| **FinBERT** | **0.8790** | **0.8409** | **0.7816** | **0.8218** | **0.9191** |
 ------------------------------------------------------------------------------

---

##  Setup

### 1. Clone the repository
```bash
git clone https://github.com/Vinodhini-03/FinancialSentimentBERT.git
cd FinancialSentimentBERT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download FinBERT checkpoint
FinBERT_best.pt is not included in the repository due to GitHub's 100MB file size limit.
Train it using the Colab notebook provided, or contact the author.

---

##  Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

##  Run Notebooks

Run in order:

01_EDA.ipynb
02_Preprocessing.ipynb
03_Models.ipynb
04_BERT.ipynb

---

##  Sample Predictions

 ------------------------------------------------------------------------------
|                           Input                                 | Prediction |
|-----------------------------------------------------------------|------------|
| "Apple cuts iPhone production forecast amid weak demand"        | 🔴 Bearish |
| "Amazon beats Q4 earnings estimates with record revenue growth" | 🟢 Bullish |
| "Federal Reserve holds interest rates steady"                   | 🔵 Neutral |
 ------------------------------------------------------------------------------

---

##  Tech Stack

 -------------------------------------------------
|    Category     |            Tools              |
|-----------------|-------------------------------|
| Language        | Python 3.11                   |
| Deep Learning   | PyTorch 2.1                   |
| Transformers    | HuggingFace Transformers 4.40 |
| NLP             | NLTK                          |
| Dashboard       | Streamlit                     |
| Visualization   | Matplotlib, Seaborn           |
| Version Control | Git, GitHub                   |
| GPU Training    | Google Colab (Tesla T4)       |
 -------------------------------------------------

---
