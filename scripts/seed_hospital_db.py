import random
import string
from datetime import datetime, timedelta
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from hospital_db_service import DB_PATH, PATIENTS_FILE, _write_json
import os
import json

def random_name():
    first_names = ["John", "Jane", "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def random_date(start, end):
    return (start + timedelta(days=random.randint(0, (end - start).days))).strftime("%Y-%m-%d")

def random_patient():
    dob = datetime.strptime("1950-01-01", "%Y-%m-%d") + timedelta(days=random.randint(0, 25550))
    return {
        "name": random_name(),
        "gender": random.choice(["M", "F"]),
        "age": random.randint(10, 99),
        "dob": dob.strftime("%Y-%m-%d"),
        "address": f"{random.randint(10,999)} {random.choice(['Maple', 'Oak', 'Pine', 'Cedar', 'Elm'])} St",
        "contact": ''.join(random.choices(string.digits, k=10)),
    }

def main():
    os.makedirs(DB_PATH, exist_ok=True)
    # Seed patients
    patients = []
    for _ in range(50):
        p = random_patient()
        p["patient_id"] = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        patients.append(p)
    _write_json(PATIENTS_FILE, patients)
    
    # Now add entries for other tables
    def add_records(file, records):
        with open(os.path.join(DB_PATH, file), 'w') as f:
            json.dump(records, f, indent=2)

    visits = []
    vitals = []
    labs = []
    diagnoses = []
    predictions = []
    notes = []

    for p in patients:
        pid = p["patient_id"]
        for _ in range(random.randint(1,3)):
            visit_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            visit = {
                "visit_id": visit_id,
                "patient_id": pid,
                "date": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1)),
                "reason": random.choice(["Fever", "Checkup", "Injury", "Diabetes", "Hypertension", "Follow-up"]),
            }
            visits.append(visit)
            # vitals for visit
            for _ in range(random.randint(2,4)):
                vitals.append({
                    "vital_id": ''.join(random.choices(string.ascii_letters + string.digits, k=12)),
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "date": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1)),
                    "bp": f"{random.randint(100,140)}/{random.randint(60,90)}",
                    "pulse": random.randint(60,110),
                    "temp": round(random.uniform(97.0,104.0),1),
                })
            # labs
            for _ in range(random.randint(1,2)):
                labs.append({
                    "lab_id": ''.join(random.choices(string.ascii_letters + string.digits, k=12)),
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "date": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1)),
                    "test": random.choice(["CBC","Glucose","HbA1c","Lipid Panel","Electrolytes"]),
                    "result": random.choice(["Normal","High","Low"]),
                })
            # diagnoses
            for _ in range(random.randint(1,2)):
                diagnoses.append({
                    "diagnosis_id": ''.join(random.choices(string.ascii_letters + string.digits, k=12)),
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "date": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1)),
                    "diagnosis": random.choice(["Hypertension","Diabetes","Infection","Healthy","Asthma"]),
                })
            # predictions
            for _ in range(random.randint(1,2)):
                predictions.append({
                    "prediction_id": ''.join(random.choices(string.ascii_letters + string.digits, k=12)),
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "date": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1)),
                    "model": random.choice(["DiseasePredictor","LOSAnalyzer","CohortTagger"]),
                    "result": random.choice(["High Risk","Low Risk","Normal"]),
                })
            # notes
            for _ in range(random.randint(1,3)):
                notes.append({
                    "note_id": ''.join(random.choices(string.ascii_letters + string.digits, k=12)),
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "date": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1)),
                    "content": random.choice([
                        "Patient recovering well.",
                        "Monitor blood pressure.",
                        "Patient needs follow-up.",
                        "No significant findings."
                    ])
                })
    add_records("visits.json", visits)
    add_records("vitals.json", vitals)
    add_records("labs.json", labs)
    add_records("diagnoses.json", diagnoses)
    add_records("predictions.json", predictions)
    add_records("notes.json", notes)

if __name__ == "__main__":
    main()
