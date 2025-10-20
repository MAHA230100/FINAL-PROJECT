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

### 2.1) Download sample datasets
```
python scripts/download_sample_data.py
```
CSV files will be stored under `data/raw/`.

### 2.2) Run basic EDA + cleaning
```
python scripts/eda_clean.py
```
Outputs go to `data/processed/`.

### 2.3) Process MIMIC-IV Clinical Data (Optional)
For real clinical data, run these scripts in order:
```bash
cd healthai/data
python extract_mimic_data.py    # Extract from compressed files
python clean_mimic_data.py      # Clean and create ML datasets
python process_mimic_data.py    # Train models
```
See `healthai/data/README_MIMIC.md` for detailed instructions.

### 2.4) Train baseline models (tabular)
- Classification (breast cancer dataset; target: `target`):
```
python -m healthai.src.models.tabular.baseline classification healthai/data/raw/classification_breast_cancer.csv --target target
```
- Regression (diabetes dataset; target: `target`):
```
python -m healthai.src.models.tabular.baseline regression healthai/data/raw/regression_diabetes.csv --target target
```
Models will be saved under `healthai/models/`.

### 2.5) Evaluate saved model
```
python -m healthai.src.models.tabular.evaluate healthai/models/baseline_classification.pkl healthai/data/raw/classification_breast_cancer.csv --target target --task classification
```

### 3) Run API
```
uvicorn src.api.main:app --reload
```
API docs at `http://localhost:8000/docs`.

### 4) Run Dashboard
```
streamlit run src/ui/dashboard.py
```

### 5) Run with Docker
From the `healthai/docker` folder:
```
docker compose up --build
```
- API: `http://localhost:8000/health` and `/docs`
- UI: `http://localhost:8501`
To smoke-test:
```
python scripts/test_docker.py
```

## Repo layout
```
src/
  api/            # FastAPI app
  ui/             # Streamlit dashboard
  data/           # Data loading & preprocessing
  models/         # ML/DL models (nlp, vision, timeseries, association, tabular)
  utils/          # IO, metrics, helpers
notebooks/        # EDA and experiments
docs/             # Architecture and design
``` 