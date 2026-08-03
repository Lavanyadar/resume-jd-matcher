"""
Resume ↔ Job Description Matcher — Streamlit App

Loads the trained Siamese GRU model (`resume_jd_gru_model.keras`) and the shared
tokenizer (`resume_jd_tokenizer.pkl`) produced by `ResumeJD_GRU.ipynb`, and predicts
whether a resume is a match for a job description.

Place these two files in the same folder as this script:
    resume_jd_gru_model.keras
    resume_jd_tokenizer.pkl
"""

import re
import pickle

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# Config — must match the values used in ResumeJD_GRU.ipynb
# ---------------------------------------------------------------------------
MODEL_PATH = "resume_jd_gru_model.keras"
TOKENIZER_PATH = "resume_jd_tokenizer.pkl"

MAX_LEN_RESUME = 300
MAX_LEN_JD = 150

st.set_page_config(page_title="Resume ↔ JD Matcher", page_icon="🧩", layout="centered")


# ---------------------------------------------------------------------------
# Text cleaning — same as clean_text() in ResumeJD_Logistic_Regression.ipynb,
# which produced the resume/jd text that every downstream notebook trains on.
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)       # strip any HTML
    text = re.sub(r"[^a-z\s]", " ", text)    # keep only letters
    text = re.sub(r"\s+", " ", text).strip() # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_gru_model():
    return load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_match(resume_text: str, jd_text: str) -> float:
    tokenizer = load_tokenizer()
    model = load_gru_model()

    resume_seq = tokenizer.texts_to_sequences([clean_text(resume_text)])
    jd_seq = tokenizer.texts_to_sequences([clean_text(jd_text)])

    resume_pad = pad_sequences(resume_seq, maxlen=MAX_LEN_RESUME, padding="post", truncating="post")
    jd_pad = pad_sequences(jd_seq, maxlen=MAX_LEN_JD, padding="post", truncating="post")

    proba = model.predict([resume_pad, jd_pad], verbose=0)[0][0]
    return float(proba)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧩 Resume ↔ Job Description Matcher")
st.caption(
    "Paste a resume and a job description below to check how well they match, "
    "using the trained Siamese GRU model."
)

col1, col2 = st.columns(2)
with col1:
    resume_input = st.text_area("Resume text", height=280, placeholder="Paste resume text here...")
with col2:
    jd_input = st.text_area("Job description text", height=280, placeholder="Paste job description here...")

threshold = st.slider("Match threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if st.button("Check Match", type="primary"):
    if not resume_input.strip() or not jd_input.strip():
        st.warning("Please enter both a resume and a job description.")
    else:
        with st.spinner("Scoring..."):
            try:
                score = predict_match(resume_input, jd_input)
            except FileNotFoundError as e:
                st.error(
                    f"Couldn't find a required file: {e.filename}. "
                    f"Make sure '{MODEL_PATH}' and '{TOKENIZER_PATH}' are in the same "
                    "folder as app.py."
                )
                st.stop()

        label = "✅ Match" if score >= threshold else "❌ No Match"
        st.subheader(label)
        st.metric("Match score", f"{score:.2%}")
        st.progress(min(max(score, 0.0), 1.0))
