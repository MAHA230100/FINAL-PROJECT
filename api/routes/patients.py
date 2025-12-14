import sys
from fastapi import APIRouter, HTTPException, Request
from .hospital_db_service import create_patient, fetch_patient, get_all_patients

router = APIRouter()

@router.get("/patients")
def list_patients():
    """List all patients"""
    return get_all_patients()

@router.post("/patients")
async def add_patient(request: Request):
    data = await request.json()
    patient = create_patient(data)
    return {"patient_id": patient["patient_id"]}

@router.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    patient = fetch_patient(patient_id)
    if not patient:
        raise HTTPException(404, detail="Patient not found")
    return patient
