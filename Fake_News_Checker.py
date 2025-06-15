import re
import streamlit as st
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer
import torch
import torch.nn.functional as F
import numpy as np
import base64
from pathlib import Path

# preprocessing 
url_re = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")
whitespace_re = re.compile(r"\s+")
label_prefix_re = re.compile(
    r'^\s*(?:fake news[:\-\–]\s*|true news[:\-\–]\s*|fake[:\-\–]\s*|true[:\-\–]\s*)',
    flags=re.IGNORECASE
)

def strip_label_prefix(s):
    return label_prefix_re.sub("", s).strip()

def clean_text(title, body):
    t = strip_label_prefix(title)
    b = strip_label_prefix(body)
    join = f"{t} {b}"
    no_urls = url_re.sub("<URL>", join)
    return whitespace_re.sub(" ", no_urls).strip()

# load model
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained("./Model")
    model = RobertaForSequenceClassification.from_pretrained("./Model")
    model.config.id2label = {0: "FAKE", 1: "REAL"}
    model.config.label2id = {"FAKE": 0, "REAL": 1}
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# change background
def set_background_with_overlay(image_file, overlay_rgba="rgba(0, 0, 0, 0.5)"):
    # read & base64‑encode your image
    img_bytes = Path(image_file).read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    st.markdown("""
        <style>
        /* Style for text input and text area */
        .stTextInput > div > div > input,
        .stTextArea > div > textarea {
            background-color: rgba(255, 255, 255, 0.85);
            border: 1px solid #ccc;
            padding: 0.5rem;
            border-radius: 10px;
            color: #000;
        }

        /* Optional: make placeholder text darker */
        ::placeholder {
            color: #666;
            opacity: 1;
        }

        /* Style label text (like "Headline", "Article text") */
        label {
            font-weight: bold;
            font-size: 1.1rem;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

set_background_with_overlay("news_detection_background.jpg", overlay_rgba="rgba(0,0,0,0.6)")

# build website app
st.title("📰 Fake News Detector")
title = st.text_input("Headline")
body  = st.text_area("Article text", height=300)

if st.button("Check"):
    text = clean_text(title, body)

    inputs = tokenizer(
        text,
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_i = int(np.argmax(probs))
    label  = model.config.id2label[pred_i]
    conf   = probs[pred_i] * 100

    st.markdown(f"**Prediction:** {'✅ REAL' if label=='REAL' else '❌ FAKE'}")
    st.markdown(f"**Confidence:** {conf:.1f}%")