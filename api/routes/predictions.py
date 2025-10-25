from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from pathlib import Path
import joblib
import numpy as np

router = APIRouter(prefix="/predict", tags=["predictions"])


class ClassificationRequest(BaseModel):
    features: List[float]


class RegressionRequest(BaseModel):
    features: List[float]


# Model paths
MODELS_DIR = Path("healthai/models")
CLASSIFY_MODEL = MODELS_DIR / "mimic_mortality_model.pkl"
CLASSIFY_SCALER = MODELS_DIR / "mimic_mortality_scaler.pkl"
CLASSIFY_FEATURES = MODELS_DIR / "mimic_mortality_features.pkl"

REGRESS_MODEL = MODELS_DIR / "mimic_los_model.pkl"
REGRESS_SCALER = MODELS_DIR / "mimic_los_scaler.pkl"
REGRESS_FEATURES = MODELS_DIR / "mimic_los_features.pkl"

# Fallback models
BASELINE_CLASSIFY = MODELS_DIR / "baseline_classification.pkl"
BASELINE_REGRESS = MODELS_DIR / "baseline_regression.pkl"


def _safe_load(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@router.post("/classify")
def predict_classify(req: ClassificationRequest):
    """Disease risk classification endpoint"""
    # Try MIMIC mortality model first
    model = _safe_load(CLASSIFY_MODEL)
    scaler = _safe_load(CLASSIFY_SCALER)
    features = _safe_load(CLASSIFY_FEATURES)
    
    if model is None or scaler is None or features is None:
        # Fallback to baseline model
        model = _safe_load(BASELINE_CLASSIFY)
        if model is None:
            return {"prediction": 0, "proba": 0.12, "note": "No classification model found; returning stub"}
        X = np.array([req.features])
        scaler = None
    else:
        # Use MIMIC model with scaling
        X = np.array([req.features])
        X = scaler.transform(X)
    
    pred = model.predict(X)[0]
    proba = None
    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception:
        proba = None
    
    return {
        "prediction": int(pred) if isinstance(pred, (np.integer,)) else pred, 
        "proba": proba,
        "model_type": "mimic_mortality" if scaler is not None else "baseline"
    }


@router.post("/regress")
def predict_regress(req: RegressionRequest):
    """Length of stay prediction endpoint"""
    # Try MIMIC LOS model first
    model = _safe_load(REGRESS_MODEL)
    scaler = _safe_load(REGRESS_SCALER)
    features = _safe_load(REGRESS_FEATURES)
    
    if model is None or scaler is None or features is None:
        # Fallback to baseline model
        model = _safe_load(BASELINE_REGRESS)
        if model is None:
            return {"prediction": 3.7, "note": "No regression model found; returning stub"}
        X = np.array([req.features])
        scaler = None
    else:
        # Use MIMIC model with scaling
        X = np.array([req.features])
        X = scaler.transform(X)
    
    pred = float(model.predict(X)[0])
    return {
        "prediction": pred,
        "model_type": "mimic_los" if scaler is not None else "baseline"
    }
