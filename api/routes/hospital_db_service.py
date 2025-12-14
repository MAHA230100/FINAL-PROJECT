import json
import os
from typing import Optional, Dict, Any, List
from uuid import uuid4

# --- Robust: Ensure /hospital_db/ exists and all .json table files exist ---
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'hospital_db')
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
