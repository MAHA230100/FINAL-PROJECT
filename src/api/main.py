from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="HealthAI API", version="0.1.0")


class ClassificationRequest(BaseModel):
    features: list[float]


class RegressionRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/classify")
def predict_classify(req: ClassificationRequest):
    # TODO: load model and run inference
    return {"risk_class": "low", "proba": 0.12}


@app.post("/predict/regress")
def predict_regress(req: RegressionRequest):
    # TODO: load model and run inference
    return {"los_days": 3.7}
