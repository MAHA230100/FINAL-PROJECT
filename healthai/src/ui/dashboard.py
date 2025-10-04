import streamlit as st
import requests
import numpy as np
import os

st.set_page_config(page_title="HealthAI Dashboard", layout="wide")

st.title("HealthAI Dashboard")

_default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
API_BASE = st.sidebar.text_input("API base URL", _default_api, key="api_base_url")


def parse_features(text: str):
	try:
		vals = [float(x.strip()) for x in text.split(",") if x.strip()]
		return vals
	except Exception:
		return None


tabs = st.tabs([
    "Classification",
    "Regression",
    "Clustering",
    "Associations",
    "Imaging (CNN)",
    "Time Series (LSTM)",
    "NLP (BERT)",
    "Translator",
    "Sentiment",
    "FAQ Chatbot",
])

with tabs[0]:
    st.subheader("Disease Risk Classification")
    features = st.text_input("Enter features (comma-separated)", key="cls_features")
    if st.button("Predict risk", key="predict_risk"):
        vals = parse_features(features)
        if not vals:
            st.error("Provide numeric features, comma-separated")
        else:
            try:
                res = requests.post(f"{API_BASE}/predict/classify", json={"features": vals}, timeout=10)
                st.json(res.json())
            except Exception as e:
                st.error(f"Request failed: {e}")

with tabs[1]:
    st.subheader("Length of Stay Prediction")
    features = st.text_input("Enter features (comma-separated)", key="reg_features")
    if st.button("Predict LOS", key="predict_los"):
        vals = parse_features(features)
        if not vals:
            st.error("Provide numeric features, comma-separated")
        else:
            try:
                res = requests.post(f"{API_BASE}/predict/regress", json={"features": vals}, timeout=10)
                st.json(res.json())
            except Exception as e:
                st.error(f"Request failed: {e}")

with tabs[2]:
    st.subheader("Patient Clustering")
    st.info("Stub: visualize clusters and profiles")

with tabs[3]:
    st.subheader("Association Rules")
    st.info("Stub: list rules with support/confidence/lift")

with tabs[4]:
    st.subheader("Imaging Diagnostics (CNN)")
    st.info("Stub: upload X-ray and show prediction")

with tabs[5]:
    st.subheader("Time Series Forecasting (LSTM)")
    st.info("Stub: upload vitals time series and forecast")

with tabs[6]:
    st.subheader("Clinical NLP (BioBERT/ClinicalBERT)")
    st.info("Stub: NER/summarization on clinical notes")

with tabs[7]:
    st.subheader("Medical Translator")
    st.info("Stub: English ↔ regional language translation")

with tabs[8]:
    st.subheader("Sentiment Analysis")
    st.info("Stub: analyze patient feedback sentiment")

with tabs[9]:
    st.subheader("FAQ Chatbot (RAG-lite)")
    try:
        from healthai.src.chatbot.faq_rag import build_default_rag
        rag = build_default_rag()
        q = st.text_input("Ask a question", key="chatbot_query")
        if st.button("Ask", key="chatbot_ask") and q.strip():
            ans = rag.answer(q.strip())
            st.write(ans)
        st.caption("Place .txt files under healthai/docs/faq to extend knowledge base.")
    except Exception as e:
        st.error(f"Chatbot failed to initialize: {e}") 