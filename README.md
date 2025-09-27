# HealthAI: End-to-End Clinical AI System

This project implements an end-to-end AI/ML system over healthcare data (tabular EHR, imaging, and text) to:
- Predict outcomes (regression)
- Classify disease risk (classification)
- Discover patient subgroups (clustering)
- Mine medical associations (association rules)
- Build/compare deep learning models (CNN, RNN/LSTM)
- Leverage pretrained clinical NLP (BioBERT/ClinicalBERT)
- Provide a healthcare chatbot (RAG)
- Provide a medical translator
- Perform sentiment analysis on patient feedback

It includes a FastAPI backend and a Streamlit dashboard.

## Quickstart

### 1) Python environment
- Python 3.10+ recommended
- Create venv and install dependencies:
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure
Edit `src/config/config.yaml` (paths, seeds, settings). Place datasets under a data folder you choose (see config).

### 3) Run API
```
uvicorn src.api.main:app --reload
```
API docs at `http://localhost:8000/docs`.

### 4) Run Dashboard
```
streamlit run src/app/dashboard.py
```

## Repo layout
```
src/
  api/            # FastAPI app
  app/            # Streamlit dashboard
  data/           # Data loading & preprocessing
  features/       # Feature engineering
  models/         # Classical ML models
  timeseries/     # LSTM, sequence models
  vision/         # CNN for imaging
  nlp/            # BERT, sentiment, translator, RAG
  association/    # Association rule mining
  utils/          # IO, metrics, helpers
notebooks/        # EDA and experiments
scripts/          # CLI entry points
tests/            # Unit tests
```

## Datasets (examples)
- MIMIC-III/IV (EHR)
- PhysioNet (vitals time series)
- NIH Chest X-ray 14 (imaging)
- Patient feedback (Kaggle or portals)
- Synthetic/anonymized when needed

## Experiment tracking
- MLflow hooks available (optional). Set tracking URI/env as needed.

## Ethics and privacy
- Use only anonymized data
- Follow HIPAA/GDPR guidelines
- Include model cards and SHAP explanations for interpretability

## Status
Scaffold and minimal stubs ready. Expand modules incrementally per course deliverables.
