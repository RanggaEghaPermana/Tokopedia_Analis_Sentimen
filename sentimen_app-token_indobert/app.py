# app.py – Analisis Sentimen Tokopedia (v2.2: Perbaikan UI & Hover Final)
import streamlit as st
import torch
import plotly.graph_objects as go
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_extras.let_it_rain import rain

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Sentimen Tokopedia",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "indoBERT_tokped"
LOTTIE_DIR = BASE_DIR / "lottie"
ICON_FILE = BASE_DIR / "tokopedia.png"

# ---------------- Custom CSS (dengan perbaikan hover & focus) ----------------
st.markdown("""
<style>
    /* Base font and theme */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    /* Main title */
    h1 {
        font-weight: 700;
        letter-spacing: -1.5px;
        color: #4A4A4A;
    }
    /* Tokopedia green color for accents */
    div[data-testid="stProgress"] div div div {
        background-color: #03AC0E !important;
    }
    /* Style dasar tombol */
    .stButton > button {
        background-color: #03AC0E !important;
        color: white !important;         /* pastikan teks putih */
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    /* Saat hover, focus atau active (klik) */
    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active {
        background-color: #028A0B !important;
        color: white !important;         /* teks tetap putih */
        transform: scale(1.05);
    }
    /* Center the placeholder for Lottie */
    div[data-testid="stVerticalBlock"] {
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Asset Loading (Cached) ----------------
@st.cache_resource
def load_model():
    """Load and cache the fine-tuned model and tokenizer."""
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).eval()
    labels = [mdl.config.id2label[i] for i in sorted(mdl.config.id2label)]
    return tok, mdl, labels

@st.cache_data
def load_lottie_file(path: Path):
    """Load and cache a Lottie JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Animasi tidak ditemukan di: {path}. Pastikan file ada dan nama file sudah benar (positive.json, neutral.json, negative.json).")
        return None

# Load assets
tok, model, LABELS = load_model()
LOTTIE_FILES = {
    "Positive": load_lottie_file(LOTTIE_DIR / "positive.json"),
    "Neutral": load_lottie_file(LOTTIE_DIR / "neutral.json"),
    "Negative": load_lottie_file(LOTTIE_DIR / "negative.json"),
}

# ---------------- Sidebar Navigation ----------------
with st.sidebar:
    st.image(str(ICON_FILE), width=80)
    st.markdown("<h1 style='font-size: 24px;'>Analisis Sentimen Ulasan</h1>", unsafe_allow_html=True)
    st.caption("Ini Cuman Demo Ya Teman Teman.")

    selected = option_menu(
        menu_title=None,
        options=["Analyzer"],
        icons=["robot"],
        menu_icon="cast",
        default_index=0,
    )
    st.info("Dibuat dengan model IndoBERT yang di-fine-tune pada ~11K ulasan produk Tokopedia.")

# ---------------- Main Interface ----------------
st.title("🤖 Analisis Sentimen Tokopedia")
st.markdown("Masukkan ulasan produk Anda di bawah ini. AI akan menganalisis apakah sentimennya positif, netral, atau negatif.")

# Create placeholders for dynamic content
gauge_placeholder = st.empty()
result_placeholder = st.empty()
user_input = st.text_area("Tulis ulasan Anda di sini...", height=150, placeholder="Contoh: Barangnya bagus banget, pengiriman super cepat!")

if st.button("Analisis Sekarang", use_container_width=True):
    if user_input:
        # --- Inference ---
        with st.spinner("🧠 AI sedang berpikir..."):
            tokens = tok(
                user_input.lower(),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            )
            with torch.no_grad():
                logits = model(**tokens).logits
                probs = torch.softmax(logits, dim=1)[0].tolist()
            idx = int(torch.argmax(logits))
            label = LABELS[idx]
            conf = probs[idx]

        # --- Display Results ---
        # 1. Gauge Chart
        color_map = {"Positive": "#03AC0E", "Neutral": "#f59e0b", "Negative": "#ef4444"}
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf * 100,
            number={'suffix': ' %', 'font': {'size': 24}},
            title={'text': f"Keyakinan Model ({label})", 'font': {'size': 18, 'color': 'gray'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color_map.get(label, "gray")},
                'steps': [
                    {'range': [0, 50], 'color': '#F8F8F8'},
                    {'range': [50, 100], 'color': '#F0F0F0'}
                ],
            }
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
        gauge_placeholder.plotly_chart(fig, use_container_width=True)

        # 2. Lottie Animation and Text
        col1, col2 = result_placeholder.columns([1, 2])
        animation = LOTTIE_FILES.get(label)
        if animation:
            with col1:
                st_lottie(animation, height=150, key=label.lower())

        with col2:
            st.markdown(f"### Prediksi: **{label}**")
            st.markdown(f"#### Keyakinan: `{conf:.1%}`")
            # Empathetic response
            responses = {
                "Positive": "🎉 **Luar biasa!** Terima kasih atas ulasan positifnya. Senang Anda menyukainya!",
                "Neutral": "🙏 **Terima kasih atas masukannya.** Kami akan terus berupaya menjadi lebih baik.",
                "Negative": "😥 **Mohon maaf atas pengalaman kurang baik.** Kami akan segera menindaklanjuti masukan Anda."
            }
            st.warning(responses[label])

        # Fun little extra for positive reviews
        if label == "Positive" and conf > 0.9:
            rain(emoji="🎉", font_size=54, falling_speed=5, animation_length="1s")
    else:
        st.error("⚠️ Harap masukkan ulasan terlebih dahulu.")
