#Imports
import streamlit as st
import torch
import torch.nn as nn
import pickle
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Page Config
st.set_page_config(
    page_title = "Financial Sentiment Predictor",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

#Custom CSS
st.markdown("""
<style>
    /* Fix radio button and sidebar text visibility */
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #e0e0e0 !important;
    }
    div[role="radiogroup"] label {
        color: #e0e0e0 !important;
    }
    div[role="radiogroup"] label p {
        color: #e0e0e0 !important;
    }
    /* Main background */
    .stApp {
        background-color: #1a1a2e;
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #16213e;
        border-right: 1px solid #0f3460;
        color: #e0e0e0 !important;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f3460, #16213e);
        padding: 25px 30px;
        border-radius: 12px;
        border-left: 5px solid #ffffff;
        margin-bottom: 25px;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #a0a0b0;
        margin: 5px 0 0 0;
        font-size: 0.95rem;
    }

    /* Prediction card */
    .pred-card {
        background-color: #16213e;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #0f3460;
        margin-bottom: 20px;
    }

    /* Sentiment badges */
    .badge-bearish {
        background-color: #e94560;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-bullish {
        background-color: #2ecc71;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #3498db;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
    }

    /* Metric cards */
    .metric-card {
        background-color: #16213e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #0f3460;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #a0a0b0;
        margin-top: 4px;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #0f3460;
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 8px;
        font-size: 1rem;
    }

    /* Button */
    .stButton > button {
        background-color: #00b4d8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #c0392b;
        color: white;
    }

    /* Section headers */
    .section-header {
        color: #00b4d8;
        font-size: 1.1rem;
        font-weight: 700;
        border-bottom: 1px solid #0f3460;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }

    /* Table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .comparison-table th {
        background-color: #0f3460;
        color: #ffffff;
        padding: 10px;
        text-align: center;
        font-size: 0.85rem;
    }
    .comparison-table td {
        padding: 10px;
        text-align: center;
        border-bottom: 1px solid #0f3460;
        color: #e0e0e0;
        font-size: 0.85rem;
    }
    .comparison-table tr:hover {
        background-color: #0f3460;
    }
    .winner-row {
        background-color: #1a3a1a;
        font-weight: 700;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

#Model Architectures
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout, pad_idx):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded        = self.dropout(self.embedding(x))
        _, (hidden, _)  = self.lstm(embedded)
        hidden          = self.dropout(hidden[-1])
        return self.fc(hidden)
    
    #Paths
import os
BASE_DIR   = r"D:\FinancialSentimentBERT"
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

#Load LSTM Artifacts
@st.cache_resource
def load_lstm():
    with open(os.path.join(MODEL_DIR, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
        config = json.load(f)

    model = SentimentLSTM(
        vocab_size  = config['VOCAB_SIZE'],
        embed_dim   = 128,
        hidden_dim  = 256,
        num_classes = 3,
        num_layers  = 2,
        dropout     = 0.3,
        pad_idx     = config['PAD_IDX']
    )
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'LSTM_best.pt'),
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model, word2idx, config

#Load FinBERT
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model     = AutoModelForSequenceClassification.from_pretrained(
        'ProsusAI/finbert',
        num_labels              = 3,
        ignore_mismatched_sizes = True
    )
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'FinBERT_best.pt'),
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model, tokenizer

#Preprocessing 
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
keep_words = {'up', 'down', 'not', 'no', 'nor', 'against', 'below', 'above', 'under', 'over'}
stop_words = stop_words - keep_words

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\$([A-Za-z]+)', r'\1', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def light_clean(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text, word2idx, max_len=32):
    tokens   = text.split()
    sequence = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    sequence = sequence[:max_len]
    sequence = sequence + [word2idx['<PAD>']] * (max_len - len(sequence))
    return sequence

#Prediction Functions
def predict_lstm(text, model, word2idx):
    cleaned  = clean_text(text)
    sequence = text_to_sequence(cleaned, word2idx)
    tensor   = torch.tensor([sequence], dtype=torch.long)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).squeeze().numpy()
    return probs

def predict_finbert(text, model, tokenizer):
    cleaned  = light_clean(text)
    encoding = tokenizer(
        cleaned,
        max_length     = 64,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )
    with torch.no_grad():
        output = model(
            input_ids      = encoding['input_ids'],
            attention_mask = encoding['attention_mask']
        )
        probs = torch.softmax(output.logits, dim=1).squeeze().numpy()
    return probs

#Sidebar 
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <span style='font-size:2.5rem;'>📈</span>
        <h2 style='color:#00b4d8; margin:5px 0;'>FinSentiment</h2>
        <p style='color:#a0a0b0; font-size:0.8rem;'>Financial News Sentiment Predictor</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🤖 Select Model</div>", unsafe_allow_html=True)
    selected_model = st.radio(
        "",
        options    = ["LSTM (Baseline)", "FinBERT (Transformer)"],
        index      = 1,
        label_visibility = "collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Dataset Statistics</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>9,535</div>
            <div class='metric-label'>Train Samples</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>2,388</div>
            <div class='metric-label'>Valid Samples</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🏆 Model Performance</div>", unsafe_allow_html=True)

    if "FinBERT" in selected_model:
        st.markdown("""
        <div class='metric-card' style='margin-bottom:10px;'>
            <div class='metric-value'>88.09%</div>
            <div class='metric-label'>Accuracy</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>0.8409</div>
            <div class='metric-label'>Macro F1</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='metric-card' style='margin-bottom:10px;'>
            <div class='metric-value'>80.23%</div>
            <div class='metric-label'>Accuracy</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>0.7355</div>
            <div class='metric-label'>Macro F1</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📌 Sentiment Guide</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.85rem; color:#a0a0b0; line-height:2;'>
        🔴 <span style='color:#e94560; font-weight:700;'>Bearish</span> — Negative outlook<br>
        🟢 <span style='color:#2ecc71; font-weight:700;'>Bullish</span> — Positive outlook<br>
        🔵 <span style='color:#3498db; font-weight:700;'>Neutral</span> — No clear direction
    </div>
    """, unsafe_allow_html=True)

    # Main Header
st.markdown("""
<div class='main-header'>
    <h1>📈 Financial News Sentiment Predictor</h1>
    <p>Analyze financial tweets and news headlines using Deep Learning & FinBERT</p>
</div>
""", unsafe_allow_html=True)

#Prediction Section
label_names  = ['Bearish', 'Bullish', 'Neutral']
label_colors = ['#e94560', '#2ecc71', '#3498db']
label_icons  = ['🔴', '🟢', '🔵']
badge_class  = ['badge-bearish', 'badge-bullish', 'badge-neutral']

st.markdown("<div class='section-header'>💬 Enter Financial News or Tweet</div>",
            unsafe_allow_html=True)

user_input = st.text_area(
    "",
    placeholder = "e.g. Apple cuts iPhone production forecast amid weak demand...",
    height      = 120,
    label_visibility = "collapsed"
)

predict_btn = st.button("🔍 Predict Sentiment")

if predict_btn:
    if not user_input.strip():
        st.warning("⚠ Please enter a financial headline or tweet.")
    else:
        with st.spinner("Analyzing sentiment..."):
            if "FinBERT" in selected_model:
                model, tokenizer = load_finbert()
                probs = predict_finbert(user_input, model, tokenizer)
            else:
                model, word2idx, config = load_lstm()
                probs = predict_lstm(user_input, model, word2idx)

        pred_idx    = int(np.argmax(probs))
        pred_label  = label_names[pred_idx]
        pred_conf   = probs[pred_idx] * 100
        pred_color  = label_colors[pred_idx]
        pred_icon   = label_icons[pred_idx]
        pred_badge  = badge_class[pred_idx]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📊 Prediction Result</div>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"""
            <div class='pred-card' style='text-align:center;'>
                <div style='font-size:3rem;'>{pred_icon}</div>
                <div class='{pred_badge}' style='margin:10px 0;'>{pred_label}</div>
                <div style='color:#a0a0b0; font-size:0.85rem; margin-top:10px;'>Confidence</div>
                <div style='color:{pred_color}; font-size:2rem; font-weight:800;'>{pred_conf:.1f}%</div>
                <div style='color:#a0a0b0; font-size:0.75rem; margin-top:10px;'>
                    Model: {"FinBERT" if "FinBERT" in selected_model else "LSTM"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='pred-card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#16213e')
            ax.set_facecolor('#16213e')

            bars = ax.barh(label_names, probs * 100, color=label_colors, height=0.5)

            for bar, prob in zip(bars, probs):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{prob*100:.1f}%', va='center', color='white', fontsize=11,
                        fontweight='bold')

            ax.set_xlim(0, 115)
            ax.set_xlabel('Confidence (%)', color='#a0a0b0')
            ax.tick_params(colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#0f3460')
            ax.spines['bottom'].set_color('#0f3460')
            for label in ax.get_yticklabels():
                label.set_color('white')
                label.set_fontsize(11)

            plt.title('Probability Distribution', color='white', fontsize=12,
                      fontweight='bold', pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

        #Input Summary
        st.markdown(f"""
        <div class='pred-card' style='margin-top:10px;'>
            <div style='color:#a0a0b0; font-size:0.8rem;'>📝 Input Text</div>
            <div style='color:#e0e0e0; font-size:0.95rem; margin-top:5px;'>"{user_input}"</div>
        </div>
        """, unsafe_allow_html=True)
        
        #Model Comparison Section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🏆 Model Comparison</div>",
            unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <table class='comparison-table'>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Macro F1</th>
            <th>Bearish F1</th>
            <th>Bullish F1</th>
            <th>Neutral F1</th>
        </tr>
        <tr>
            <td>SimpleRNN</td>
            <td>0.4380</td>
            <td>0.3336</td>
            <td>0.2196</td>
            <td>0.1828</td>
            <td>0.5983</td>
        </tr>
        <tr>
            <td>GRU</td>
            <td>0.7956</td>
            <td>0.7339</td>
            <td>0.6257</td>
            <td>0.7139</td>
            <td>0.8623</td>
        </tr>
        <tr>
            <td>LSTM</td>
            <td>0.8023</td>
            <td>0.7355</td>
            <td>0.6417</td>
            <td>0.6899</td>
            <td>0.8749</td>
        </tr>
        <tr class='winner-row'>
            <td>🏆 FinBERT</td>
            <td>0.8790</td>
            <td>0.8409</td>
            <td>0.7816</td>
            <td>0.8218</td>
            <td>0.9191</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#16213e')
    ax.set_facecolor('#16213e')

    models   = ['SimpleRNN', 'GRU', 'LSTM', 'FinBERT']
    macro_f1 = [0.3336, 0.7339, 0.7355, 0.8409]
    colors   = ['#e94560', '#5bc0de', '#5cb85c', '#f0ad4e']

    bars = ax.bar(models, macro_f1, color=colors, width=0.5)

    for bar, val in zip(bars, macro_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', color='white',
                fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Macro F1', color='#a0a0b0')
    ax.set_title('Macro F1 — All Models', color='white',
                 fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#0f3460')
    ax.spines['bottom'].set_color('#0f3460')
    for label in ax.get_xticklabels():
        label.set_color('white')
    for label in ax.get_yticklabels():
        label.set_color('#a0a0b0')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

#Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#a0a0b0; font-size:0.8rem;
            border-top:1px solid #0f3460; padding-top:15px;'>
    Financial News Sentiment Predictor — Built with PyTorch & HuggingFace Transformers
    <br>Models: SimpleRNN | LSTM | GRU | FinBERT (ProsusAI/finbert)
</div>
""", unsafe_allow_html=True)

