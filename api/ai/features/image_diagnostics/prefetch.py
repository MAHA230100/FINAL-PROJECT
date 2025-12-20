"""Image Diagnostics - Prefetch"""
from typing import Dict, Any
from ....routes.hospital_db_service import fetch_patient_comprehensive

async def prefetch_image_data(patient_id: str, image_type: str = None) -> Dict[str, Any]:
    patient_data = fetch_patient_comprehensive(patient_id)
    if not patient_data:
        return {}
    
    return {
        "patient_id": patient_id,
        "patient_name": patient_data.get("name"),
        "age": patient_data.get("age"),
        "gender": patient_data.get("gender"),
        "image_type": image_type or "X-ray",
        "previous_diagnoses": patient_data.get("diagnoses", [])
    }
