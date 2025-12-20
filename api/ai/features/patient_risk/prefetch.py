"""
Patient Risk Assessment Feature - Data Prefetch
"""

from typing import Dict, Any
from ....routes.hospital_db_service import fetch_patient_comprehensive


def safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


async def prefetch_patient_risk_data(patient_id: str) -> Dict[str, Any]:
    """
    Prefetch all required data for patient risk assessment.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Dictionary with patient data
    """
    # Fetch comprehensive patient data
    patient_data = fetch_patient_comprehensive(patient_id)
    
    if not patient_data:
        return {}
    
    # Extract relevant fields with proper type conversion
    return {
        "patient_id": patient_id,
        "name": patient_data.get("name"),
        "age": safe_int(patient_data.get("age")),
        "gender": patient_data.get("gender"),
        "vitals_bp": safe_int(patient_data.get("vitals_bp")),
        "vitals_hr": safe_int(patient_data.get("vitals_hr")),
        "previous_admissions": safe_int(patient_data.get("previous_admissions", 0)),
        "comorbidities": patient_data.get("comorbidities", []),
        "lab_results": patient_data.get("lab_results"),
        "admission_type": patient_data.get("admission_type"),
        "diagnoses": patient_data.get("diagnoses", []),
        "vitals": patient_data.get("vitals", []),
    }
