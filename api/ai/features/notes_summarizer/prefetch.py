"""Notes Summarizer - Prefetch"""
from typing import Dict, Any
from ....routes.hospital_db_service import fetch_patient_comprehensive

async def prefetch_notes_data(patient_id: str) -> Dict[str, Any]:
    patient_data = fetch_patient_comprehensive(patient_id)
    if not patient_data:
        return {}
    
    notes = patient_data.get("notes", [])
    # Combine all notes into single text
    notes_text = "\n\n".join([n.get("note_text", "") for n in notes if n.get("note_text")])
    
    return {
        "patient_id": patient_id,
        "notes_text": notes_text,
        "patient_name": patient_data.get("name"),
        "visit_count": len(patient_data.get("visits", []))
    }
