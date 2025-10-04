from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np

app = FastAPI(title="HealthAI API", version="0.1.0")


class ClassificationRequest(BaseModel):
    features: list[float]


class RegressionRequest(BaseModel):
    features: list[float]


MODELS_DIR = Path("healthai/models")
CLASSIFY_MODEL = MODELS_DIR / "baseline_classification.pkl"
REGRESS_MODEL = MODELS_DIR / "baseline_regression.pkl"


@app.get("/health")
def health():
    return {"status": "ok"}


def _safe_load(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@app.post("/predict/classify")
def predict_classify(req: ClassificationRequest):
    model = _safe_load(CLASSIFY_MODEL)
    if model is None:
        # Stub if model not available
        return {"risk_class": "low", "proba": 0.12, "note": "baseline_classification.pkl not found; returning stub"}
    X = np.array([req.features])
    pred = model.predict(X)[0]
    proba = None
    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception:
        proba = None
    return {"prediction": int(pred) if isinstance(pred, (np.integer,)) else pred, "proba": proba}


@app.post("/predict/regress")
def predict_regress(req: RegressionRequest):
    model = _safe_load(REGRESS_MODEL)
    if model is None:
        return {"los_days": 3.7, "note": "baseline_regression.pkl not found; returning stub"}
    X = np.array([req.features])
    pred = float(model.predict(X)[0])
    return {"prediction": pred} 