import re
import streamlit as st
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer
import torch
import torch.nn.functional as F
import numpy as np

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

def set_background(gif_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{gif_url}") center / cover no-repeat fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

GIF_URL = "https://i.pinimg.com/originals/d4/81/f3/d481f3c72e283309071f79e01b05c06d.gif"
set_background(GIF_URL)

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