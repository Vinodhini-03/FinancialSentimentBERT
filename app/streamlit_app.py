import streamlit as st
import torch
import torch.nn as nn
import pickle
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import io
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import requests

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSent AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #F8FAFC !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #0A2540 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1200px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0A2540 !important;
    border-right: none !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; font-size: 0.9rem !important; }
[data-testid="stSidebar"] .stRadio label p { color: #CBD5E1 !important; }
div[role="radiogroup"] label { color: #CBD5E1 !important; }
div[role="radiogroup"] label p { color: #CBD5E1 !important; }

/* ── Sidebar nav item selected ── */
div[role="radiogroup"] label:has(input:checked) p {
    color: #38BDF8 !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #0A2540 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #1a3a5c !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(10,37,64,0.18) !important;
}

/* ── Text areas ── */
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: #0A2540 !important;
    padding: 12px !important;
}
.stTextArea textarea:focus {
    border-color: #0A2540 !important;
    box-shadow: 0 0 0 3px rgba(10,37,64,0.08) !important;
}

/* ── Select box ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    color: #0A2540 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px !important;
    border-bottom: 2px solid #E2E8F0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 6px 6px 0 0 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #0A2540 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #0A2540 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #0A2540 !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* ── Divider ── */
hr { border-color: #E2E8F0 !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
# Models folder is one level up from the app folder
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

LABEL_NAMES  = ['Bearish', 'Bullish', 'Neutral']
LABEL_COLORS = ['#E63946', '#00A67E', '#4361EE']
LABEL_ICONS  = ['📉', '📈', '➡️']
LABEL_BG     = ['#FEF2F2', '#F0FDF4', '#EEF2FF']
LABEL_BORDER = ['#FECACA', '#BBF7D0', '#C7D2FE']

MODEL_METRICS = {
    'SimpleRNN': {'accuracy': 0.4380, 'macro_f1': 0.3336, 'bearish_f1': 0.2196, 'bullish_f1': 0.1828, 'neutral_f1': 0.5983},
    'LSTM':      {'accuracy': 0.8023, 'macro_f1': 0.7355, 'bearish_f1': 0.6417, 'bullish_f1': 0.6899, 'neutral_f1': 0.8749},
    'GRU':       {'accuracy': 0.7956, 'macro_f1': 0.7339, 'bearish_f1': 0.6257, 'bullish_f1': 0.7139, 'neutral_f1': 0.8623},
    'FinBERT':   {'accuracy': 0.8790, 'macro_f1': 0.8409, 'bearish_f1': 0.7816, 'bullish_f1': 0.8218, 'neutral_f1': 0.9191},
}

EXAMPLE_TWEETS = [
    "Apple cuts iPhone production forecast amid weak demand",
    "Tesla stock surges after record quarterly deliveries",
    "Fed holds interest rates steady amid inflation concerns",
    "Oil prices plunge as OPEC increases output unexpectedly",
    "Amazon reports strong earnings, beats analyst expectations",
]

# ─── Model Architectures ─────────────────────────────────────────────────────
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        embedded       = self.dropout(self.embedding(x))
        _, hidden      = self.rnn(embedded)
        hidden         = self.dropout(hidden[-1])
        return self.fc(hidden)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout, pad_idx):
        super().__init__()
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

class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru       = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        embedded       = self.dropout(self.embedding(x))
        _, hidden      = self.gru(embedded)
        hidden         = self.dropout(hidden[-1])
        return self.fc(hidden)

# ─── Load Artifacts ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vocab_config():
    with open(os.path.join(MODEL_DIR, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    return word2idx, config

@st.cache_resource(show_spinner=False)
def load_rnn(config):
    model = SentimentRNN(vocab_size=config['VOCAB_SIZE'], embed_dim=128, hidden_dim=256,
                         num_classes=3, num_layers=2, dropout=0.3, pad_idx=config['PAD_IDX'])
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'SimpleRNN_best.pt'), map_location='cpu'))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_lstm(config):
    model = SentimentLSTM(vocab_size=config['VOCAB_SIZE'], embed_dim=128, hidden_dim=256,
                          num_classes=3, num_layers=2, dropout=0.3, pad_idx=config['PAD_IDX'])
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'LSTM_best.pt'), map_location='cpu'))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_gru(config):
    model = SentimentGRU(vocab_size=config['VOCAB_SIZE'], embed_dim=128, hidden_dim=256,
                         num_classes=3, num_layers=2, dropout=0.3, pad_idx=config['PAD_IDX'])
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'GRU_best.pt'), map_location='cpu'))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model     = AutoModelForSequenceClassification.from_pretrained(
        'ProsusAI/finbert', num_labels=3, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'FinBERT_best.pt'), map_location='cpu'))
    model.eval()
    return model, tokenizer

# ─── Preprocessing ───────────────────────────────────────────────────────────
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
keep_words = {'up','down','not','no','nor','against','below','above','under','over'}
stop_words = stop_words - keep_words

def clean_text(text):
    text   = text.lower()
    text   = re.sub(r'http\S+|www\S+', '', text)
    text   = re.sub(r'@\w+', '', text)
    text   = re.sub(r'\$([A-Za-z]+)', r'\1', text)
    text   = re.sub(r'#(\w+)', r'\1', text)
    text   = re.sub(r'[^a-z\s]', '', text)
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def light_clean(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def text_to_sequence(text, word2idx, max_len=32):
    tokens   = text.split()
    sequence = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    sequence = sequence[:max_len]
    sequence = sequence + [word2idx['<PAD>']] * (max_len - len(sequence))
    return sequence

# ─── Prediction ──────────────────────────────────────────────────────────────
def predict_rnn_model(text, model, word2idx):
    cleaned  = clean_text(text)
    sequence = text_to_sequence(cleaned, word2idx)
    tensor   = torch.tensor([sequence], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()
    return probs

def predict_finbert_model(text, model, tokenizer):
    cleaned  = light_clean(text)
    encoding = tokenizer(cleaned, max_length=64, padding='max_length',
                         truncation=True, return_tensors='pt')
    with torch.no_grad():
        probs = torch.softmax(
            model(input_ids=encoding['input_ids'],
                  attention_mask=encoding['attention_mask']).logits,
            dim=1).squeeze().numpy()
    return probs

def run_prediction(text, selected_model, word2idx, config):
    t0 = time.time()
    if selected_model == 'FinBERT':
        model, tokenizer = load_finbert()
        probs = predict_finbert_model(text, model, tokenizer)
    elif selected_model == 'LSTM':
        model = load_lstm(config)
        probs = predict_rnn_model(text, model, word2idx)
    elif selected_model == 'GRU':
        model = load_gru(config)
        probs = predict_rnn_model(text, model, word2idx)
    else:
        model = load_rnn(config)
        probs = predict_rnn_model(text, model, word2idx)
    elapsed = (time.time() - t0) * 1000
    return probs, elapsed

# ─── Rule-Based Override Layer ────────────────────────────────────────────────
BEARISH_PHRASES = [
    "strategic alternatives", "covenant flexibility", "stepping down",
    "explore options", "restructuring", "bridge round", "right-size",
    "monitoring inventory", "rating watch", "lender discussions",
    "guidance revision", "lowered expectations", "cautious outlook",
    "internal review", "accounting practices", "workforce reduction",
    "credit watch", "debt restructuring", "liquidity concerns",
    "cash flow pressure", "dividend cut", "impairment charge",
    "pursue other opportunities", "pursue other interests",
    "step down", "steps down", "resigns", "resignation",
    "departing", "leaving the company", "exit the role",
    "explore strategic", "strategic review", "under review",
    "lender covenant", "covenant breach", "debt covenant",
    "going concern", "bankruptcy", "chapter 11", "insolvency",
    "workforce cut", "headcount reduction", "layoffs",
    "profit warning", "revenue shortfall", "missed expectations",
    "inventory levels closely", "monitoring its inventory",
    "rating review", "credit agencies placed", "placed on watch",
    "on watch for", "watch for a potential",
    "recently revised forecasts", "recently revised",
    "revised forecasts", "revised guidance", "lowered guidance",
    "weak demand", "slowing demand", "softening demand",
    "right size", "right sizing", "cost reduction plan",
    "significant workforce", "reduce its workforce",
]
BULLISH_PHRASES = [
    "record revenue", "record profit", "beats expectations",
    "raised guidance", "strong earnings", "record deliveries",
    "buyback acceleration", "share repurchase", "record backlog",
    "margin expansion", "free cash flow growth", "market share gain",
    "ahead of schedule", "breakeven ahead", "above expectations",
    "accretive to earnings", "accretive to eps", "immediately accretive",
    "free cash flow conversion", "fcf conversion",
    "not needed to draw", "has not needed to draw",
    "surpassed industry", "surpassed benchmarks",
    "highest level in", "highest in five", "highest in three",
    "beat analyst", "beats analyst", "beating analyst",
    "exceeded expectations", "ahead of analyst", "ahead of forecast",
    "ahead of projection", "record quarterly", "record annual",
    "record earnings", "record orders", "record sales",
    "surged after", "surges after", "jumped after",
    "beating wall street", "beat wall street",
]

def rule_based_check(text):
    text_lower = text.lower()
    for phrase in BEARISH_PHRASES:
        if phrase in text_lower:
            return 0, phrase  # Bearish index=0
    for phrase in BULLISH_PHRASES:
        if phrase in text_lower:
            return 1, phrase  # Bullish index=1
    return None, None

# ─── Groq LLM Explanation ────────────────────────────────────────────────────
def get_llm_explanation(text, sentiment, confidence, rule_triggered=False):
    if not GROQ_API_KEY:
        return None
    low_confidence  = confidence < 65.0
    rule_context    = f"IMPORTANT: Despite what the model predicted, a financial domain rule detected this as {sentiment}. Explain why it IS {sentiment} — do not second-guess this." if rule_triggered else ""
    confidence_note = f"Note: Model confidence is low ({confidence:.1f}%). Carefully analyze the true sentiment." if low_confidence else ""

    prompt = f"""You are a senior financial analyst. A deep learning model classified this headline.

Headline: "{text}"
Final Sentiment: {sentiment}
{rule_context}
{confidence_note}

Explain WHY this headline is {sentiment}. Be direct and confident — do not say it is ambiguous or neutral if the sentiment says otherwise.

Respond ONLY in this exact format with no extra text:

EXPLANATION:
[2-3 sentences explaining why this is {sentiment}. Be specific about the financial meaning of the phrases used.]

KEY SIGNALS:
[2-3 bullet points. Each: emoji + phrase from headline + reason why it signals {sentiment}]

INVESTOR IMPLICATION:
[1 sentence on what this means for investors]

Under 120 words total."""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3
            },
            timeout=15
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def render_llm_explanation(explanation, sentiment, confidence, rule_triggered=False):
    import streamlit.components.v1 as components
    idx       = LABEL_NAMES.index(sentiment)
    color     = LABEL_COLORS[idx]
    bg        = LABEL_BG[idx]
    border    = LABEL_BORDER[idx]
    low_conf  = confidence < 65.0

    warning_html = ""
    if low_conf:
        warning_html += """
        <div style="background:#FEF9C3;border:1px solid #FDE047;border-radius:8px;
                    padding:0.6rem 1rem;margin-bottom:0.8rem;font-size:13px;color:#854D0E;">
            ⚠️ <strong>Low confidence prediction.</strong> The AI analysis below may reveal a different sentiment.
        </div>"""
    if rule_triggered:
        warning_html += f"""
        <div style="background:{bg};border:1px solid {border};border-radius:8px;
                    padding:0.6rem 1rem;margin-bottom:0.8rem;font-size:13px;color:{color};">
            🔎 <strong>Smart Override Active:</strong> A known financial phrase pattern was detected and used to refine the prediction.
        </div>"""

    # Parse sections
    sections = {"EXPLANATION": "", "KEY SIGNALS": "", "INVESTOR IMPLICATION": ""}
    current  = None
    for line in explanation.split('\n'):
        line = line.strip()
        if not line:
            continue
        matched = False
        for key in sections:
            if line.upper().startswith(key):
                current = key
                rest    = line[len(key):].lstrip(':').strip()
                if rest:
                    sections[key] += rest + " "
                matched = True
                break
        if not matched and current:
            sections[current] += line + " "

    exp_html  = f"<div style='margin-bottom:0.8rem;'><div style='font-size:11px;font-weight:700;color:#94A3B8;letter-spacing:0.08em;margin-bottom:4px;'>EXPLANATION</div><div style='font-size:14px;color:#374151;line-height:1.7;'>{sections['EXPLANATION'].strip()}</div></div>" if sections['EXPLANATION'].strip() else ""
    sig_html  = f"<div style='background:#F8FAFC;border-radius:8px;padding:0.8rem;margin-bottom:0.8rem;'><div style='font-size:11px;font-weight:700;color:#94A3B8;letter-spacing:0.08em;margin-bottom:4px;'>KEY SIGNALS</div><div style='font-size:13px;color:#374151;line-height:1.8;'>{sections['KEY SIGNALS'].strip()}</div></div>" if sections['KEY SIGNALS'].strip() else ""
    imp_html  = f"<div style='background:{bg};border-radius:8px;padding:0.8rem;'><div style='font-size:11px;font-weight:700;color:{color};letter-spacing:0.08em;margin-bottom:4px;'>INVESTOR IMPLICATION</div><div style='font-size:13px;color:#374151;line-height:1.7;'>{sections['INVESTOR IMPLICATION'].strip()}</div></div>" if sections['INVESTOR IMPLICATION'].strip() else ""

    full_html = f"""
    <div style="font-family:'DM Sans',sans-serif;background:#ffffff;border:1.5px solid {border};
                border-radius:14px;padding:1.2rem;margin-top:1rem;border-left:4px solid {color};">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
            <span style="font-size:18px;">🤖</span>
            <span style="font-size:15px;font-weight:800;color:#0A2540;">AI Analysis</span>
            <span style="margin-left:auto;background:{bg};color:{color};font-size:11px;
                         font-weight:700;padding:3px 10px;border-radius:50px;">Powered by LLaMA 3.3 70B</span>
        </div>
        {warning_html}
        {exp_html}
        {sig_html}
        {imp_html}
    </div>"""

    components.html(full_html, height=380, scrolling=False)

# ─── UI Components ───────────────────────────────────────────────────────────
def render_result_card(text, probs, model_name, show_text=True):
    idx    = int(np.argmax(probs))
    label  = LABEL_NAMES[idx]
    conf   = probs[idx] * 100
    color  = LABEL_COLORS[idx]
    icon   = LABEL_ICONS[idx]
    bg     = LABEL_BG[idx]
    border = LABEL_BORDER[idx]

    if show_text:
        st.markdown(f"""
        <div style='background:#fff; border:1.5px solid #E2E8F0; border-radius:14px;
                    padding:1.2rem 1.4rem; margin-bottom:1rem; border-left:4px solid {color};'>
            <div style='font-size:0.78rem; color:#94A3B8; margin-bottom:0.4rem;
                        font-family:DM Sans,sans-serif;'>📝 Input</div>
            <div style='font-size:0.92rem; color:#0A2540; font-weight:500;
                        font-family:DM Sans,sans-serif; margin-bottom:1rem;'>"{text}"</div>
            <div style='display:flex; align-items:center; gap:1rem; flex-wrap:wrap;'>
                <div style='background:{bg}; border:1.5px solid {border}; border-radius:50px;
                            padding:0.4rem 1.2rem; display:inline-flex; align-items:center; gap:0.5rem;'>
                    <span style='font-size:1.1rem;'>{icon}</span>
                    <span style='font-size:1rem; font-weight:700; color:{color};
                                 font-family:DM Sans,sans-serif;'>{label}</span>
                </div>
                <div style='font-size:0.88rem; color:#64748B; font-family:DM Sans,sans-serif;'>
                    Confidence: <strong style='color:{color};'>{conf:.1f}%</strong>
                    &nbsp;·&nbsp; Model: <strong>{model_name}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:{bg}; border:1.5px solid {border}; border-radius:12px;
                    padding:1.2rem; text-align:center;'>
            <div style='font-size:2rem;'>{icon}</div>
            <div style='font-size:1.3rem; font-weight:800; color:{color};
                        font-family:DM Sans,sans-serif; margin:0.3rem 0;'>{label}</div>
            <div style='font-size:2rem; font-weight:800; color:{color};
                        font-family:DM Sans,sans-serif;'>{conf:.1f}%</div>
            <div style='font-size:0.78rem; color:#94A3B8; margin-top:0.3rem;'>Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    return idx, conf

def render_confidence_chart(probs, title="Probability Distribution"):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    bars = ax.barh(LABEL_NAMES, probs * 100, color=LABEL_COLORS, height=0.45,
                   edgecolor='none')
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{prob*100:.1f}%', va='center', color='#0A2540',
                fontsize=10, fontweight='600', fontfamily='sans-serif')
    ax.set_xlim(0, 118)
    ax.set_xlabel('Confidence (%)', color='#64748B', fontsize=9)
    ax.set_title(title, color='#0A2540', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#0A2540', labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('#F8FAFC')
    ax.grid(axis='x', color='#E2E8F0', linewidth=0.7, linestyle='--')
    ax.set_axisbelow(True)
    for label in ax.get_yticklabels():
        label.set_color('#0A2540')
        label.set_fontweight('600')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def render_summary_pie(counts, title="Sentiment Distribution"):
    labels = [n for n, c in zip(LABEL_NAMES, counts) if c > 0]
    sizes  = [c for c in counts if c > 0]
    colors = [LABEL_COLORS[i] for i, c in enumerate(counts) if c > 0]
    if not sizes:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#FFFFFF')
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=2))
    for t in texts:
        t.set_color('#0A2540'); t.set_fontsize(10); t.set_fontweight('600')
    for at in autotexts:
        at.set_color('white'); at.set_fontsize(10); at.set_fontweight('700')
    ax.set_title(title, color='#0A2540', fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:2rem 1.2rem 1.5rem 1.2rem; border-bottom:1px solid #1E3A5F;'>
        <div style='font-size:2.2rem; margin-bottom:0.4rem;'>📈</div>
        <div style='font-family:Playfair Display,serif; font-size:1.4rem; font-weight:800;
                    color:#F8FAFC; line-height:1.2;'>FinSent AI</div>
        <div style='font-size:0.78rem; color:#94A3B8; margin-top:0.3rem;
                    font-family:DM Sans,sans-serif;'>Financial Sentiment Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem; color:#64748B; font-weight:700; letter-spacing:0.08em; padding:0 1.2rem; font-family:DM Sans,sans-serif;'>NAVIGATION</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

    page = st.radio("", ["🏠  Home", "🔍  Predict", "📊  Model Comparison", "📖  About", "👩‍💻  Contact"],
                    label_visibility="collapsed")

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:1px solid #1E3A5F; padding-top:1.2rem; margin:0 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:0 0.5rem;'>
        <div style='font-size:0.72rem; color:#64748B; font-weight:700; letter-spacing:0.08em;
                    margin-bottom:0.8rem; font-family:DM Sans,sans-serif;'>MODEL PERFORMANCE</div>
        <div style='display:flex; flex-direction:column; gap:0.5rem;'>
            <div style='background:#1E3A5F; border-radius:8px; padding:0.6rem 0.8rem;
                        display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-size:0.82rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>FinBERT</span>
                <span style='font-size:0.82rem; font-weight:700; color:#34D399; font-family:DM Sans,sans-serif;'>88.09%</span>
            </div>
            <div style='background:#1E3A5F; border-radius:8px; padding:0.6rem 0.8rem;
                        display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-size:0.82rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>LSTM</span>
                <span style='font-size:0.82rem; font-weight:700; color:#60A5FA; font-family:DM Sans,sans-serif;'>80.23%</span>
            </div>
            <div style='background:#1E3A5F; border-radius:8px; padding:0.6rem 0.8rem;
                        display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-size:0.82rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>GRU</span>
                <span style='font-size:0.82rem; font-weight:700; color:#60A5FA; font-family:DM Sans,sans-serif;'>79.56%</span>
            </div>
            <div style='background:#1E3A5F; border-radius:8px; padding:0.6rem 0.8rem;
                        display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-size:0.82rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>SimpleRNN</span>
                <span style='font-size:0.82rem; font-weight:700; color:#F87171; font-family:DM Sans,sans-serif;'>43.80%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:0 0.5rem;'>
        <div style='font-size:0.72rem; color:#64748B; font-weight:700; letter-spacing:0.08em;
                    margin-bottom:0.6rem; font-family:DM Sans,sans-serif;'>SENTIMENT GUIDE</div>
        <div style='font-size:0.82rem; color:#94A3B8; line-height:2; font-family:DM Sans,sans-serif;'>
            📉 <span style='color:#F87171; font-weight:600;'>Bearish</span> — Negative outlook<br>
            📈 <span style='color:#34D399; font-weight:600;'>Bullish</span> — Positive outlook<br>
            ➡️ <span style='color:#818CF8; font-weight:600;'>Neutral</span> — No clear direction
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Load vocab & config once ─────────────────────────────────────────────────
word2idx, config = load_vocab_config()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style='padding:3.5rem 0 2rem 0;'>
        <div style='font-size:0.82rem; font-weight:700; letter-spacing:0.12em; color:#4361EE;
                    margin-bottom:1rem; font-family:DM Sans,sans-serif;'>DEEP LEARNING · NLP · FINANCE</div>
        <div style='font-family:Playfair Display,serif; font-size:3rem; font-weight:800;
                    color:#0A2540; line-height:1.15; max-width:700px;'>
            Predict Financial Market<br>
            <span style='color:#4361EE;'>Sentiment</span> with AI
        </div>
        <div style='font-size:1.05rem; color:#64748B; margin-top:1.2rem; max-width:580px;
                    line-height:1.7; font-family:DM Sans,sans-serif;'>
            Classify financial tweets and news headlines as
            <strong style='color:#E63946;'>Bearish</strong>,
            <strong style='color:#00A67E;'>Bullish</strong>, or
            <strong style='color:#4361EE;'>Neutral</strong> using four
            deep learning models — SimpleRNN, LSTM, GRU, and FinBERT.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("4", "Deep Learning Models", "#4361EE"),
        ("88.09%", "Best Model Accuracy", "#00A67E"),
        ("11,923", "Training Samples", "#0A2540"),
        ("3", "Sentiment Classes", "#E63946"),
    ]
    for col, (val, label, color) in zip([col1,col2,col3,col4], stats):
        with col:
            st.markdown(f"""
            <div style='background:#ffffff; border:1.5px solid #E2E8F0; border-radius:14px;
                        padding:1.4rem 1.2rem; text-align:center; border-top:3px solid {color};'>
                <div style='font-family:Playfair Display,serif; font-size:1.9rem; font-weight:800;
                            color:{color};'>{val}</div>
                <div style='font-size:0.8rem; color:#64748B; margin-top:0.3rem;
                            font-family:DM Sans,sans-serif;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div style='font-family:Playfair Display,serif; font-size:1.6rem; font-weight:800;
                color:#0A2540; margin-bottom:1.2rem;'>What You Can Do</div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    features = [
        ("🔍", "Single Prediction", "Enter any financial headline or tweet and instantly get sentiment classification with confidence scores.", "#4361EE"),
        ("📋", "Multi-Tweet Batch", "Analyze multiple headlines at once. Each tweet is predicted individually with its own result card.", "#00A67E"),
        ("📁", "File Upload", "Upload a CSV or TXT file of headlines. Get a full results table with downloadable predictions.", "#E63946"),
    ]
    for col, (icon, title, desc, color) in zip([fc1,fc2,fc3], features):
        with col:
            st.markdown(f"""
            <div style='background:#ffffff; border:1.5px solid #E2E8F0; border-radius:14px;
                        padding:1.5rem; height:100%;'>
                <div style='font-size:1.8rem; margin-bottom:0.8rem;'>{icon}</div>
                <div style='font-size:1rem; font-weight:700; color:#0A2540; margin-bottom:0.5rem;
                            font-family:DM Sans,sans-serif;'>{title}</div>
                <div style='font-size:0.85rem; color:#64748B; line-height:1.6;
                            font-family:DM Sans,sans-serif;'>{desc}</div>
                <div style='margin-top:1rem; height:3px; background:{color};
                            border-radius:2px; width:40px;'></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    # Model lineup
    st.markdown("""
    <div style='font-family:Playfair Display,serif; font-size:1.6rem; font-weight:800;
                color:#0A2540; margin-bottom:1.2rem;'>Model Lineup</div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    models_info = [
        ("SimpleRNN", "Baseline", "43.80%", "0.3336", "#94A3B8", "Vanishing gradient limits performance on financial text."),
        ("LSTM", "Advanced RNN", "80.23%", "0.7355", "#60A5FA", "Long-range memory captures complex financial context effectively."),
        ("GRU", "Efficient RNN", "79.56%", "0.7339", "#818CF8", "Competitive with LSTM using fewer parameters."),
        ("FinBERT", "🏆 Champion", "88.09%", "0.8409", "#34D399", "Finance-domain BERT. Best accuracy and F1 across all classes."),
    ]
    for col, (name, tag, acc, f1, color, desc) in zip([mc1,mc2,mc3,mc4], models_info):
        with col:
            st.markdown(f"""
            <div style='background:#ffffff; border:1.5px solid #E2E8F0; border-radius:14px;
                        padding:1.2rem; border-top:3px solid {color};'>
                <div style='font-size:0.72rem; font-weight:700; color:{color}; letter-spacing:0.08em;
                            margin-bottom:0.4rem; font-family:DM Sans,sans-serif;'>{tag.upper()}</div>
                <div style='font-size:1.05rem; font-weight:800; color:#0A2540; margin-bottom:0.8rem;
                            font-family:DM Sans,sans-serif;'>{name}</div>
                <div style='display:flex; gap:0.6rem; margin-bottom:0.8rem;'>
                    <div style='background:#F1F5F9; border-radius:6px; padding:0.3rem 0.6rem; flex:1; text-align:center;'>
                        <div style='font-size:1rem; font-weight:800; color:{color};'>{acc}</div>
                        <div style='font-size:0.68rem; color:#94A3B8;'>Accuracy</div>
                    </div>
                    <div style='background:#F1F5F9; border-radius:6px; padding:0.3rem 0.6rem; flex:1; text-align:center;'>
                        <div style='font-size:1rem; font-weight:800; color:{color};'>{f1}</div>
                        <div style='font-size:0.68rem; color:#94A3B8;'>Macro F1</div>
                    </div>
                </div>
                <div style='font-size:0.78rem; color:#64748B; line-height:1.5;
                            font-family:DM Sans,sans-serif;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0A2540,#1E3A5F); border-radius:16px;
                padding:2.5rem; text-align:center;'>
        <div style='font-family:Playfair Display,serif; font-size:1.8rem; font-weight:800;
                    color:#F8FAFC; margin-bottom:0.6rem;'>Ready to Analyze?</div>
        <div style='font-size:0.95rem; color:#94A3B8; margin-bottom:1.5rem;
                    font-family:DM Sans,sans-serif;'>
            Head over to the Predict page and try it with your own financial headlines.
        </div>
        <div style='display:inline-block; background:#4361EE; color:white; padding:0.7rem 2rem;
                    border-radius:8px; font-weight:700; font-family:DM Sans,sans-serif;
                    font-size:0.95rem;'>
            🔍 Go to Predict →
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Predict":
    st.markdown("""
    <div style='padding:2rem 0 1.5rem 0;'>
        <div style='font-family:Playfair Display,serif; font-size:2.2rem; font-weight:800; color:#0A2540;'>
            Sentiment Predictor
        </div>
        <div style='font-size:0.95rem; color:#64748B; margin-top:0.4rem; font-family:DM Sans,sans-serif;'>
            Analyze financial headlines using deep learning models
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    sel_col1, sel_col2 = st.columns([2, 3])
    with sel_col1:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#0A2540; margin-bottom:0.4rem; font-family:DM Sans,sans-serif;'>SELECT MODEL</div>", unsafe_allow_html=True)
        selected_model = st.selectbox("", ["FinBERT", "LSTM", "GRU", "SimpleRNN"],
                                      label_visibility="collapsed")
    with sel_col2:
        m = MODEL_METRICS[selected_model]
        st.markdown(f"""
        <div style='background:#ffffff; border:1.5px solid #E2E8F0; border-radius:10px;
                    padding:0.8rem 1.2rem; display:flex; gap:2rem; align-items:center; margin-top:1.5rem;'>
            <div style='text-align:center;'>
                <div style='font-size:1.1rem; font-weight:800; color:#00A67E;'>{m['accuracy']*100:.2f}%</div>
                <div style='font-size:0.72rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>Accuracy</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.1rem; font-weight:800; color:#4361EE;'>{m['macro_f1']:.4f}</div>
                <div style='font-size:0.72rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>Macro F1</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.1rem; font-weight:800; color:#E63946;'>{m['bearish_f1']:.4f}</div>
                <div style='font-size:0.72rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>Bearish F1</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:1.1rem; font-weight:800; color:#00A67E;'>{m['bullish_f1']:.4f}</div>
                <div style='font-size:0.72rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>Bullish F1</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝  Single Headline", "📋  Multi-Tweet Batch", "📁  File Upload"])

    # ── Tab 1: Single ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        # Example chips
        st.markdown("<div style='font-size:0.78rem; color:#94A3B8; margin-bottom:0.5rem; font-family:DM Sans,sans-serif;'>💡 Try an example:</div>", unsafe_allow_html=True)
        ex_cols = st.columns(len(EXAMPLE_TWEETS))
        clicked_example = None
        for col, ex in zip(ex_cols, EXAMPLE_TWEETS):
            with col:
                if st.button(ex[:35] + "…", key=f"ex_{ex[:10]}"):
                    clicked_example = ex

        user_input = st.text_area(
            "Enter financial headline or tweet",
            value=clicked_example if clicked_example else "",
            placeholder="e.g. Apple cuts iPhone production forecast amid weak demand...",
            height=110,
            key="single_input"
        )

        if st.button("🔍  Analyze Sentiment", key="single_btn"):
            if not user_input.strip():
                st.warning("⚠️ Please enter a financial headline or tweet.")
            else:
                with st.spinner("Analyzing..."):
                    probs, elapsed = run_prediction(user_input.strip(), selected_model, word2idx, config)

                # Rule-based override check
                rule_idx, rule_phrase = rule_based_check(user_input.strip())
                rule_triggered = False
                if rule_idx is not None:
                    # Hard override — force the rule-matched class to win clearly
                    new_probs = np.array([0.05, 0.05, 0.05])
                    new_probs[rule_idx] = 0.85
                    probs = new_probs
                    rule_triggered = True

                idx        = int(np.argmax(probs))
                sentiment  = LABEL_NAMES[idx]
                confidence = probs[idx] * 100

                st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
                r1, r2 = st.columns([1, 2])
                with r1:
                    render_result_card(user_input.strip(), probs, selected_model, show_text=False)
                    st.markdown(f"""
                    <div style='text-align:center; font-size:0.78rem; color:#94A3B8;
                                margin-top:0.5rem; font-family:DM Sans,sans-serif;'>
                        ⚡ {elapsed:.0f}ms inference time
                    </div>
                    """, unsafe_allow_html=True)
                with r2:
                    render_confidence_chart(probs)

                # LLM Explanation
                if GROQ_API_KEY:
                    with st.spinner("🤖 Generating AI analysis..."):
                        explanation = get_llm_explanation(
                            user_input.strip(), sentiment, confidence, rule_triggered)
                    if explanation:
                        render_llm_explanation(explanation, sentiment, confidence, rule_triggered)
                else:
                    st.info("💡 Add GROQ_API_KEY to .env to enable AI explanations.")

    # ── Tab 2: Multi-Tweet ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#EEF2FF; border-radius:10px; padding:0.8rem 1.1rem; margin-bottom:1rem;
                    border-left:3px solid #4361EE;'>
            <div style='font-size:0.85rem; color:#4361EE; font-weight:600; font-family:DM Sans,sans-serif;'>
                📌 Enter one headline per line. Each will be predicted individually.
            </div>
        </div>
        """, unsafe_allow_html=True)

        multi_input = st.text_area(
            "Enter multiple headlines (one per line)",
            placeholder="Apple cuts iPhone production forecast...\nTesla surges after record deliveries...\nFed holds rates steady amid inflation...",
            height=180,
            key="multi_input"
        )

        if st.button("🔍  Analyze All Tweets", key="multi_btn"):
            lines = [l.strip() for l in multi_input.strip().split('\n') if l.strip()]
            if not lines:
                st.warning("⚠️ Please enter at least one headline.")
            else:
                st.markdown(f"""
                <div style='font-size:0.9rem; font-weight:700; color:#0A2540; margin:1rem 0 0.8rem 0;
                            font-family:DM Sans,sans-serif;'>
                    Results — {len(lines)} headline{'s' if len(lines)>1 else ''} analyzed
                </div>
                """, unsafe_allow_html=True)

                all_probs = []
                counts    = [0, 0, 0]

                for i, line in enumerate(lines):
                    with st.spinner(f"Analyzing {i+1}/{len(lines)}..."):
                        probs, elapsed = run_prediction(line, selected_model, word2idx, config)
                    # Rule-based override
                    rule_idx, _ = rule_based_check(line)
                    rule_triggered = False
                    if rule_idx is not None:
                        new_probs = np.array([0.05, 0.05, 0.05])
                        new_probs[rule_idx] = 0.85
                        probs = new_probs
                        rule_triggered = True
                    all_probs.append(probs)
                    idx = int(np.argmax(probs))
                    counts[idx] += 1
                    render_result_card(line, probs, selected_model, show_text=True)
                    if rule_triggered:
                        st.markdown(
                            f"<div style='background:#EEF2FF;border-left:3px solid #4361EE;"
                            f"border-radius:6px;padding:0.4rem 0.8rem;font-size:0.78rem;"
                            f"color:#4361EE;margin-bottom:0.5rem;'>🔎 Smart Override Active</div>",
                            unsafe_allow_html=True
                        )

                # Summary
                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div style='font-size:1rem; font-weight:700; color:#0A2540; margin-bottom:0.8rem;
                            font-family:DM Sans,sans-serif;'>📊 Batch Summary</div>
                """, unsafe_allow_html=True)

                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown(f"""
                    <div style='background:#F8FAFC; border:1.5px solid #E2E8F0; border-radius:10px;
                                padding:1rem; text-align:center;'>
                        <div style='font-size:1.4rem; font-weight:800; color:#0A2540;'>{len(lines)}</div>
                        <div style='font-size:0.75rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>Total</div>
                    </div>""", unsafe_allow_html=True)
                for col, (label, count, color) in zip([s2, s3, s4],
                    [("Bearish", counts[0], "#E63946"), ("Bullish", counts[1], "#00A67E"), ("Neutral", counts[2], "#4361EE")]):
                    with col:
                        st.markdown(f"""
                        <div style='background:#F8FAFC; border:1.5px solid #E2E8F0; border-radius:10px;
                                    padding:1rem; text-align:center; border-top:3px solid {color};'>
                            <div style='font-size:1.4rem; font-weight:800; color:{color};'>{count}</div>
                            <div style='font-size:0.75rem; color:#94A3B8; font-family:DM Sans,sans-serif;'>{label}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
                pc1, pc2 = st.columns(2)
                with pc1:
                    render_summary_pie(counts, "Sentiment Distribution")
                with pc2:
                    avg_probs = np.mean(all_probs, axis=0)
                    render_confidence_chart(avg_probs, "Average Confidence")

    # ── Tab 3: File Upload ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#F0FDF4; border-radius:10px; padding:0.8rem 1.1rem; margin-bottom:1rem;
                    border-left:3px solid #00A67E;'>
            <div style='font-size:0.85rem; color:#00A67E; font-weight:600; font-family:DM Sans,sans-serif;'>
                📌 Upload a <strong>.txt</strong> (one headline per line) or <strong>.csv</strong>
                (with a column named <code>headline</code>, <code>text</code>, or <code>news</code>).
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload file", type=['csv','txt'],
                                         label_visibility="collapsed")

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    col_candidates = [c for c in df.columns
                                      if c.lower() in ['headline','text','news','tweet','sentence','content']]
                    if not col_candidates:
                        col_candidates = [df.columns[0]]
                    text_col = col_candidates[0]
                    headlines = df[text_col].dropna().astype(str).tolist()
                else:
                    content   = uploaded_file.read().decode('utf-8')
                    headlines = [l.strip() for l in content.split('\n') if l.strip()]

                st.markdown(f"""
                <div style='font-size:0.88rem; color:#64748B; margin-bottom:1rem;
                            font-family:DM Sans,sans-serif;'>
                    ✅ Loaded <strong>{len(headlines)}</strong> headlines from <code>{uploaded_file.name}</code>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"🚀  Predict All {len(headlines)} Headlines", key="file_btn"):
                    results  = []
                    counts   = [0, 0, 0]
                    progress = st.progress(0)

                    for i, headline in enumerate(headlines):
                        probs, elapsed = run_prediction(headline, selected_model, word2idx, config)
                        # Rule-based override
                        rule_idx, _ = rule_based_check(headline)
                        rule_triggered = False
                        if rule_idx is not None:
                            new_probs = np.array([0.05, 0.05, 0.05])
                            new_probs[rule_idx] = 0.85
                            probs = new_probs
                            rule_triggered = True
                        idx = int(np.argmax(probs))
                        counts[idx] += 1
                        results.append({
                            'Headline':       headline,
                            'Sentiment':      LABEL_NAMES[idx],
                            'Confidence':     f"{probs[idx]*100:.1f}%",
                            'Bearish %':      f"{probs[0]*100:.1f}",
                            'Bullish %':      f"{probs[1]*100:.1f}",
                            'Neutral %':      f"{probs[2]*100:.1f}",
                            'Smart Override': '✅ Yes' if rule_triggered else '—',
                            'Model':          selected_model,
                        })
                        progress.progress((i+1) / len(headlines))

                    progress.empty()
                    result_df = pd.DataFrame(results)

                    st.markdown("""
                    <div style='font-size:1rem; font-weight:700; color:#0A2540; margin:1rem 0 0.5rem 0;
                                font-family:DM Sans,sans-serif;'>📋 Prediction Results</div>
                    """, unsafe_allow_html=True)
                    st.dataframe(result_df, use_container_width=True, height=300)

                    # Download
                    csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️  Download Results as CSV",
                        data=csv_bytes,
                        file_name=f"finsent_results_{selected_model}.csv",
                        mime='text/csv'
                    )

                    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
                    chart_c1, chart_c2 = st.columns(2)
                    with chart_c1:
                        render_summary_pie(counts, "Sentiment Distribution")
                    with chart_c2:
                        fig, ax = plt.subplots(figsize=(6, 3.5))
                        fig.patch.set_facecolor('#FFFFFF')
                        ax.set_facecolor('#F8FAFC')
                        ax.bar(LABEL_NAMES, counts, color=LABEL_COLORS, width=0.5, edgecolor='none')
                        for i, (c, v) in enumerate(zip(ax.patches, counts)):
                            ax.text(c.get_x() + c.get_width()/2, c.get_height() + 0.1,
                                    str(v), ha='center', fontsize=11, fontweight='700', color='#0A2540')
                        ax.set_title("Count by Sentiment", color='#0A2540', fontsize=11, fontweight='bold')
                        ax.tick_params(colors='#0A2540')
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        ax.grid(axis='y', color='#E2E8F0', linewidth=0.7, linestyle='--')
                        ax.set_axisbelow(True)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Comparison":
    st.markdown("""
    <div style='padding:2rem 0 1.5rem 0;'>
        <div style='font-family:Playfair Display,serif; font-size:2.2rem; font-weight:800; color:#0A2540;'>
            Model Comparison
        </div>
        <div style='font-size:0.95rem; color:#64748B; margin-top:0.4rem; font-family:DM Sans,sans-serif;'>
            Side-by-side performance analysis of all four models
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    model_colors = ['#94A3B8', '#60A5FA', '#818CF8', '#34D399']
    for col, (name, metrics), color in zip([mc1,mc2,mc3,mc4], MODEL_METRICS.items(), model_colors):
        with col:
            is_winner = name == 'FinBERT'
            badge = "🏆 BEST MODEL" if is_winner else ""
            border_color = "#34D399" if is_winner else "#E2E8F0"
            shadow = "box-shadow:0 4px 20px rgba(52,211,153,0.15);" if is_winner else ""
            rows_html = (
                f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #F1F5F9;'><span style='font-size:0.78rem;color:#94A3B8;'>Accuracy</span><span style='font-size:0.78rem;font-weight:700;color:{color};'>{metrics['accuracy']*100:.2f}%</span></div>"
                f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #F1F5F9;'><span style='font-size:0.78rem;color:#94A3B8;'>Macro F1</span><span style='font-size:0.78rem;font-weight:700;color:{color};'>{metrics['macro_f1']:.4f}</span></div>"
                f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #F1F5F9;'><span style='font-size:0.78rem;color:#94A3B8;'>Bearish F1</span><span style='font-size:0.78rem;font-weight:700;color:#E63946;'>{metrics['bearish_f1']:.4f}</span></div>"
                f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #F1F5F9;'><span style='font-size:0.78rem;color:#94A3B8;'>Bullish F1</span><span style='font-size:0.78rem;font-weight:700;color:#00A67E;'>{metrics['bullish_f1']:.4f}</span></div>"
                f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;'><span style='font-size:0.78rem;color:#94A3B8;'>Neutral F1</span><span style='font-size:0.78rem;font-weight:700;color:#4361EE;'>{metrics['neutral_f1']:.4f}</span></div>"
            )
            badge_html = f"<div style='font-size:0.72rem;color:#34D399;font-weight:700;letter-spacing:0.08em;margin-bottom:0.3rem;'>{badge}</div>" if badge else ""
            st.markdown(
                f"<div style='background:#ffffff;border:1.5px solid {border_color};border-radius:14px;padding:1.2rem;border-top:3px solid {color};{shadow}'>"
                f"{badge_html}"
                f"<div style='font-size:1.05rem;font-weight:800;color:#0A2540;margin-bottom:0.8rem;'>{name}</div>"
                f"{rows_html}</div>",
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2)
    model_names   = list(MODEL_METRICS.keys())
    chart_colors  = model_colors

    with ch1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8FAFC')
        vals = [MODEL_METRICS[m]['accuracy']*100 for m in model_names]
        bars = ax.bar(model_names, vals, color=chart_colors, width=0.5, edgecolor='none')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', color='#0A2540', fontsize=10, fontweight='700')
        ax.set_ylim(0, 105)
        ax.set_title('Accuracy Comparison', color='#0A2540', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', color='#64748B', fontsize=9)
        ax.tick_params(colors='#0A2540')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.grid(axis='y', color='#E2E8F0', linewidth=0.7, linestyle='--')
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with ch2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8FAFC')
        vals = [MODEL_METRICS[m]['macro_f1'] for m in model_names]
        bars = ax.bar(model_names, vals, color=chart_colors, width=0.5, edgecolor='none')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.4f}', ha='center', color='#0A2540', fontsize=10, fontweight='700')
        ax.set_ylim(0, 1.05)
        ax.set_title('Macro F1 Comparison', color='#0A2540', fontsize=12, fontweight='bold')
        ax.set_ylabel('Macro F1 Score', color='#64748B', fontsize=9)
        ax.tick_params(colors='#0A2540')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.grid(axis='y', color='#E2E8F0', linewidth=0.7, linestyle='--')
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Per-class F1 grouped bar
    st.markdown("""
    <div style='font-size:1rem; font-weight:700; color:#0A2540; margin-bottom:0.8rem;
                font-family:DM Sans,sans-serif;'>Per-Class F1 Score Breakdown</div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    x       = np.arange(len(model_names))
    width   = 0.22
    classes = ['bearish_f1', 'bullish_f1', 'neutral_f1']
    clabels = ['Bearish', 'Bullish', 'Neutral']
    ccolors = LABEL_COLORS

    for i, (cls, clabel, ccolor) in enumerate(zip(classes, clabels, ccolors)):
        vals = [MODEL_METRICS[m][cls] for m in model_names]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=clabel, color=ccolor,
                      edgecolor='none', alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.2f}', ha='center', fontsize=8, fontweight='600', color='#0A2540')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color='#0A2540', fontsize=11, fontweight='600')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('F1 Score', color='#64748B', fontsize=9)
    ax.set_title('Per-Class F1 Score by Model', color='#0A2540', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#0A2540')
    ax.legend(loc='upper left', framealpha=0.9)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(axis='y', color='#E2E8F0', linewidth=0.7, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Live comparison
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:1rem; font-weight:700; color:#0A2540; margin-bottom:0.5rem;
                font-family:DM Sans,sans-serif;'>🔴 Live — Compare All 4 Models on Your Input</div>
    """, unsafe_allow_html=True)

    live_input = st.text_area("Enter a headline to compare all models",
                              placeholder="e.g. Amazon reports record profits this quarter...",
                              height=90, key="live_compare")
    if st.button("⚡  Run All 4 Models", key="compare_btn"):
        if not live_input.strip():
            st.warning("⚠️ Please enter a headline.")
        else:
            cols = st.columns(4)
            all_results = {}
            for col, mname in zip(cols, model_names):
                with st.spinner(f"Running {mname}..."):
                    probs, elapsed = run_prediction(live_input.strip(), mname, word2idx, config)
                all_results[mname] = (probs, elapsed)
                idx   = int(np.argmax(probs))
                with col:
                    st.markdown(f"""
                    <div style='background:#fff; border:1.5px solid #E2E8F0; border-radius:12px;
                                padding:1.1rem; text-align:center; border-top:3px solid {model_colors[list(MODEL_METRICS.keys()).index(mname)]};'>
                        <div style='font-size:0.78rem; font-weight:700; color:#94A3B8; margin-bottom:0.4rem;
                                    font-family:DM Sans,sans-serif;'>{mname}</div>
                        <div style='font-size:1.6rem;'>{LABEL_ICONS[idx]}</div>
                        <div style='font-size:1rem; font-weight:800; color:{LABEL_COLORS[idx]};
                                    margin:0.3rem 0; font-family:DM Sans,sans-serif;'>{LABEL_NAMES[idx]}</div>
                        <div style='font-size:1.4rem; font-weight:800; color:{LABEL_COLORS[idx]};
                                    font-family:DM Sans,sans-serif;'>{probs[idx]*100:.1f}%</div>
                        <div style='font-size:0.72rem; color:#94A3B8; margin-top:0.2rem;
                                    font-family:DM Sans,sans-serif;'>{elapsed:.0f}ms</div>
                    </div>
                    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖  About":
    st.markdown("""
    <div style='padding:2rem 0 1.5rem 0;'>
        <div style='font-family:Playfair Display,serif; font-size:2.2rem; font-weight:800; color:#0A2540;'>
            About FinSent AI
        </div>
        <div style='font-size:0.95rem; color:#64748B; margin-top:0.4rem; font-family:DM Sans,sans-serif;'>
            Financial news sentiment classification using deep learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#ffffff; border:1.5px solid #E2E8F0; border-radius:14px; padding:1.8rem; margin-bottom:1rem;'>
        <div style='font-size:0.9rem; color:#64748B; line-height:1.9; font-family:DM Sans,sans-serif;'>
            FinSent AI classifies financial news headlines and tweets into
            <strong style='color:#E63946;'>Bearish</strong>,
            <strong style='color:#00A67E;'>Bullish</strong>, or
            <strong style='color:#4361EE;'>Neutral</strong> sentiment using four deep learning models —
            <strong>SimpleRNN</strong>, <strong>LSTM</strong>, <strong>GRU</strong>, and
            <strong>FinBERT</strong> (ProsusAI). Models were trained on 11,923 financial samples from
            Financial PhraseBank and Twitter Financial News. Class imbalance was handled via weighted
            cross-entropy loss. FinBERT achieved the best accuracy of <strong>88.09%</strong> with
            a Macro F1 of <strong>0.8409</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key findings as native columns
    st.markdown("<div style='font-family:Playfair Display,serif; font-size:1.2rem; font-weight:800; color:#0A2540; margin-bottom:0.8rem;'>Key Findings</div>", unsafe_allow_html=True)
    kf1, kf2, kf3, kf4 = st.columns(4)
    findings = [
        ("#E63946", "#FEF2F2", "SimpleRNN Failed", "Vanishing gradient — Macro F1 only 0.33"),
        ("#00A67E", "#F0FDF4", "Learning Rate Critical", "lr=3e-4 boosted LSTM F1 from 0.26 → 0.71"),
        ("#4361EE", "#EEF2FF", "LSTM ≈ GRU", "Only 0.002 Macro F1 difference between them"),
        ("#34D399", "#F0FDF4", "FinBERT Dominates", "Domain pre-training gives ~8% accuracy gain"),
    ]
    for col, (color, bg, title, desc) in zip([kf1,kf2,kf3,kf4], findings):
        with col:
            st.markdown(
                f"<div style='background:{bg};border-radius:10px;padding:1rem;"
                f"border-left:3px solid {color};height:100%;'>"
                f"<div style='font-size:0.85rem;font-weight:700;color:#0A2540;margin-bottom:0.3rem;'>{title}</div>"
                f"<div style='font-size:0.78rem;color:#64748B;'>{desc}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Tech stack
    st.markdown("<div style='font-family:Playfair Display,serif; font-size:1.2rem; font-weight:800; color:#0A2540; margin-bottom:0.8rem;'>Tech Stack</div>", unsafe_allow_html=True)
    tech_cols = st.columns(6)
    techs = [
        ("🔥", "PyTorch"), ("🤗", "HuggingFace"), ("🐍", "Python"),
        ("📊", "Pandas/NumPy"), ("📈", "Matplotlib"), ("🚀", "Streamlit"),
    ]
    for col, (icon, name) in zip(tech_cols, techs):
        with col:
            st.markdown(
                f"<div style='background:#ffffff;border:1.5px solid #E2E8F0;border-radius:12px;"
                f"padding:1rem;text-align:center;'>"
                f"<div style='font-size:1.5rem;'>{icon}</div>"
                f"<div style='font-size:0.82rem;font-weight:700;color:#0A2540;margin-top:0.4rem;'>{name}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CONTACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👩‍💻  Contact":
    st.markdown("""
    <div style='padding:2rem 0 1.5rem 0;'>
        <div style='font-family:Playfair Display,serif; font-size:2.2rem; font-weight:800; color:#0A2540;'>
            Contact
        </div>
        <div style='font-size:0.95rem; color:#64748B; margin-top:0.4rem; font-family:DM Sans,sans-serif;'>
            Get in touch or explore the project
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style='background:#ffffff; border:1.5px solid #E2E8F0; border-radius:16px;
                    padding:2.5rem; display:flex; flex-direction:column; gap:1rem;'>
            <a href='mailto:vinodhinisugumar03@gmail.com'
               style='background:#0A2540; color:#F8FAFC; padding:1rem 1.5rem;
                      border-radius:10px; text-decoration:none; font-weight:600;
                      font-family:DM Sans,sans-serif; font-size:0.95rem; display:block;
                      text-align:center;'>
                ✉️ &nbsp; vinodhinisugumar03@gmail.com
            </a>
            <a href='https://github.com/Vinodhini-03' target='_blank'
               style='background:#0A2540; color:#F8FAFC; padding:1rem 1.5rem;
                      border-radius:10px; text-decoration:none; font-weight:600;
                      font-family:DM Sans,sans-serif; font-size:0.95rem; display:block;
                      text-align:center;'>
                🐙 &nbsp; github.com/Vinodhini-03
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#F8FAFC; border:1.5px solid #E2E8F0; border-radius:14px;
                padding:1.2rem; text-align:center;'>
        <div style='font-size:0.85rem; color:#64748B; font-family:DM Sans,sans-serif;'>
            🚀 <strong style='color:#0A2540;'>FinSent AI</strong> &nbsp;·&nbsp;
            Built with PyTorch · HuggingFace · Streamlit &nbsp;·&nbsp;
            <a href='https://github.com/Vinodhini-03' target='_blank'
               style='color:#4361EE; text-decoration:none; font-weight:600;'>
               github.com/Vinodhini-03
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)