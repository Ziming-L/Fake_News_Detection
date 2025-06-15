import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

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

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Model", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("Model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("📰 Fake News Detector")
title = st.text_input("Headline")
body  = st.text_area("Article text")

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
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
    
    pred_idx = int(torch.argmax(logits, dim=-1))
    labels = model.config.id2label
    label = labels[pred_idx]
    conf = probs[pred_idx] * 100

    st.markdown(f"**Prediction:** {'✅ REAL' if label=='REAL' else '❌ FAKE'}")
    st.markdown(f"**Confidence:** {conf:.1f}%")