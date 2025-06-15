import re
import streamlit as st
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer
import torch
import torch.nn.functional as F
import numpy as np

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
def load_trainer():
    tokenizer = RobertaTokenizer.from_pretrained("./Model")
    model = RobertaForSequenceClassification.from_pretrained("./Model")
    model.config.id2label = {0: "FAKE", 1: "REAL"}
    model.config.label2id = {"FAKE": 0, "REAL": 1}
    return Trainer(model=model, tokenizer=tokenizer)

trainer = load_trainer()
tokenizer = trainer.tokenizer

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
        return_tensors="np"
    )

    dummy_dataset = {"input_ids": inputs["input_ids"],
                     "attention_mask": inputs["attention_mask"]}
    
    outputs = trainer.predict(dummy_dataset)
    logits  = outputs.predictions
    probs   = F.softmax(torch.from_numpy(logits), dim=-1).numpy().squeeze()
    pred_i  = int(np.argmax(probs))
    label   = trainer.model.config.id2label[pred_i]
    conf    = probs[pred_i] * 100

    st.markdown(f"**Prediction:** {'✅ REAL' if label=='REAL' else '❌ FAKE'}")
    st.markdown(f"**Confidence:** {conf:.1f}%")