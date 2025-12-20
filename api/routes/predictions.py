from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import joblib
import numpy as np

router = APIRouter(prefix="/predict", tags=["predictions"])


class PredictionRequest(BaseModel):
    # Richards Features (14)
    age: int = 30
    gender: str = "Female"
    admission_type: str = "Emergency"
    admission_location: str = "Emergency Room"
    insurance: str = "Private"
    language: str = "English"
    marital_status: str = "Single"
    drg_type: str = "DRG-A"
    comorbidities: str = "None"
    lab_results: str = "Normal"
    vitals_bp: int = 120
    vitals_hr: int = 75
    previous_admissions: int = 0
    risk_score: float = 0.5
    
    # Legacy support
    features: Optional[List[float]] = None


# Model paths
models_base = Path("data/models")
MODELS_DIR = models_base
CLASSIFY_MODEL = MODELS_DIR / "classification/best_model.pkl"
CLASSIFY_SCALER = MODELS_DIR / "classification/scaler.pkl"
CLASSIFY_FEATURES = MODELS_DIR / "classification/features.pkl"

REGRESS_MODEL = MODELS_DIR / "regression/best_model.pkl"
REGRESS_SCALER = MODELS_DIR / "regression/scaler.pkl"
REGRESS_FEATURES = MODELS_DIR / "regression/features.pkl"

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
def predict_classify(req: PredictionRequest):
    """Disease risk classification endpoint (LLM Powered)"""
    try:
        from ..services.llm_service import llm_service
        if llm_service.is_active():
            # Construct a rich prompt using all 14 features
            prompt = (
                "Predict disease risk and mortality probability for this patient based on 14 clinical features:\n"
                f"- Demographics: {req.age}yo {req.gender}, {req.marital_status}, {req.language} speaker\n"
                f"- Admission: {req.admission_type} from {req.admission_location} (Insurance: {req.insurance})\n"
                f"- Clinical: Comorbidities: {req.comorbidities}, Labs: {req.lab_results}, DRG: {req.drg_type}\n"
                f"- Vitals: BP {req.vitals_bp}, HR {req.vitals_hr}\n"
                f"- History: {req.previous_admissions} prev admissions. Calculated Risk Score: {req.risk_score}\n"
                "\n"
                "Task: Predict risk of Readmission or Mortality.\n"
                "Return Valid JSON: {\"prediction\": 0 or 1, \"proba\": 0.0 to 1.0, \"reasoning\": \"brief explanation\"}"
            )
            
            response = llm_service.generate_response(prompt).strip()
            
            # Clean response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            import json
            data = json.loads(response)
            
            return {
                "prediction": int(data.get("prediction", 0)),
                "proba": float(data.get("proba", 0.5)),
                "model_type": "gemini_llm_14_features",
                "reasoning": data.get("reasoning", "AI Analysis")
            }
            
    except Exception as e:
        print(f"LLM Prediction failed: {e}")

    # Final Stub Fallback
    return {"prediction": 0, "proba": 0.12, "note": "LLM failed"}


@router.post("/regress")
def predict_regress(req: PredictionRequest):
    """Length of stay prediction endpoint (LLM Powered)"""
    try:
        from ..services.llm_service import llm_service
        if llm_service.is_active():
            prompt = (
                "Predict hospital Length of Stay (days) for patient:\n"
                f"- Demographics: {req.age}yo {req.gender}\n"
                f"- Admission: {req.admission_type}, {req.comorbidities}\n"
                f"- Vitals: BP {req.vitals_bp}, HR {req.vitals_hr}\n"
                "Return Valid JSON: {\"prediction\": number_of_days}"
            )
            response = llm_service.generate_response(prompt).strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            import json
            data = json.loads(response)
            return {
                "prediction": float(data.get("prediction", 4.0)),
                "model_type": "gemini_llm_14_features"
            }
    except Exception as e:
        print(f"LLM Regression failed: {e}")

    return {"prediction": 3.7, "note": "LLM failed"}
