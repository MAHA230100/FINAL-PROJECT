import json
import os
from typing import Optional, Dict, Any, List
from uuid import uuid4

# --- Robust: Ensure /hospital_db/ exists and all .json table files exist ---
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/hospital_db')
TABLES = [
    'patients.json',
    'visits.json',
    'vitals.json',
    'labs.json',
    'diagnoses.json',
    'predictions.json',
    'notes.json',
]
os.makedirs(DB_PATH, exist_ok=True)
for fname in TABLES:
    target_file = os.path.join(DB_PATH, fname)
    if not os.path.exists(target_file):
        with open(target_file, 'w') as f:
            json.dump([], f)

PATIENTS_FILE = os.path.join(DB_PATH, 'patients.json')


def _read_json(filename: str) -> List[Dict[str, Any]]:
    with open(filename, 'r') as f:
        return json.load(f)

def _write_json(filename: str, data: List[Dict[str, Any]]):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def create_patient(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    patients = _read_json(PATIENTS_FILE)
    patient_id = str(uuid4())
    patient_record = {
        "patient_id": patient_id,
        **patient_data
    }
    patients.append(patient_record)
    _write_json(PATIENTS_FILE, patients)
    return patient_record

def fetch_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    patients = _read_json(PATIENTS_FILE)
    for patient in patients:
        if patient["patient_id"] == patient_id:
            return patient
    return None

def update_patient(patient_id: str, new_data: Dict[str, Any]) -> bool:
    patients = _read_json(PATIENTS_FILE)
    for i, patient in enumerate(patients):
        if patient["patient_id"] == patient_id:
            patients[i].update(new_data)
            _write_json(PATIENTS_FILE, patients)
            return True
    return False

def delete_patient(patient_id: str) -> bool:
    patients = _read_json(PATIENTS_FILE)
    new_patients = [p for p in patients if p["patient_id"] != patient_id]
    if len(new_patients) == len(patients):
        return False  # No patient was deleted
    _write_json(PATIENTS_FILE, new_patients)
    return True

def get_all_patients() -> List[Dict[str, Any]]:
    """Return all patients."""
    return _read_json(PATIENTS_FILE)

def fetch_patient_comprehensive(patient_id: str) -> Optional[Dict[str, Any]]:
    """Fetch all related records for a patient (Visits, Vitals, Labs, etc)."""
    patient = fetch_patient(patient_id)
    if not patient:
        return None
        
    try:
        # Read related tables
        visits = [v for v in _read_json(os.path.join(DB_PATH, 'visits.json')) if v.get('patient_id') == patient_id]
        vitals = [v for v in _read_json(os.path.join(DB_PATH, 'vitals.json')) if v.get('patient_id') == patient_id]
        labs = [v for v in _read_json(os.path.join(DB_PATH, 'labs.json')) if v.get('patient_id') == patient_id]
        diagnoses = [v for v in _read_json(os.path.join(DB_PATH, 'diagnoses.json')) if v.get('patient_id') == patient_id]
        notes = [v for v in _read_json(os.path.join(DB_PATH, 'notes.json')) if v.get('patient_id') == patient_id]
        
        # Attach to patient object
        patient['visits'] = visits
        patient['vitals_history'] = vitals
        patient['lab_results'] = labs
        patient['diagnoses'] = diagnoses
        patient['clinical_notes'] = notes
        
        # Update flat vital fields with latest data if available
        if vitals:
            # Simple assumption: data generated in order or appended
            latest = vitals[-1]
            patient['vitals_bp'] = latest.get('bp')
            patient['vitals_hr'] = latest.get('pulse')
            patient['temperature'] = latest.get('temp')
            
    except Exception as e:
        print(f"Error fetching comprehensive data: {e}")
        # Return basic patient if error occurs in sub-tables
        
    return patient

PREDICTIONS_FILE = os.path.join(DB_PATH, 'predictions.json')

def save_prediction(patient_id: str, tool_name: str, result: Dict[str, Any]) -> bool:
    """Save an AI tool prediction/result."""
    try:
        preds = _read_json(PREDICTIONS_FILE)
        record = {
            "id": str(uuid4()),
            "patient_id": patient_id,
            "tool_name": tool_name,
            "result": result,
            "timestamp": str(uuid4()) # Placeholder, ideally use datetime.now().isoformat()
        }
        # Ideally import datetime
        import datetime
        record["timestamp"] = datetime.datetime.now().isoformat()
        
        preds.append(record)
        _write_json(PREDICTIONS_FILE, preds)
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

# For demonstration and testing
if __name__ == "__main__":
    # Example create
    pat = create_patient({"name": "Test Patient", "age": 55})
    print("Created patient:", pat)
    fid = pat['patient_id']
    # Example fetch
    print("Fetched patient:", fetch_patient(fid))
    # Example update
    if update_patient(fid, {"age": 56}):
        print("Updated patient:", fetch_patient(fid))
    # Example delete
    if delete_patient(fid):
        print(f"Deleted patient {fid}")
    else:
        print("Delete failed.")
