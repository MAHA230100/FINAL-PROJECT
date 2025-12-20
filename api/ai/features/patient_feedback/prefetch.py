"""Patient Feedback - Prefetch"""
from typing import Dict, Any
from ....routes.hospital_db_service import fetch_patient_comprehensive

async def prefetch_feedback_data(patient_id: str, feedback_text: str = None) -> Dict[str, Any]:
    patient_data = fetch_patient_comprehensive(patient_id)
    if not patient_data:
        return {"patient_id": patient_id, "feedback_text": feedback_text or ""}
    
    return {
        "patient_id": patient_id,
        "patient_name": patient_data.get("name"),
        "feedback_text": feedback_text or "",
        "recent_visits": len(patient_data.get("visits", []))
    }
