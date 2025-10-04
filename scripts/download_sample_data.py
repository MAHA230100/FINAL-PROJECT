from __future__ import annotations
import zipfile
from pathlib import Path
import pandas as pd

# Paths
RAW_DIR = Path("healthai/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data/mimic_demo")   # <-- put your extracted demo CSVs here
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_tables() -> dict[str, pd.DataFrame]:
    """
    Load all CSVs from the local MIMIC demo folder into pandas DataFrames.
    """
    tables = {}
    for fname in DATA_DIR.rglob("*.csv"):
        name = fname.stem.lower()
        df = pd.read_csv(fname)
        tables[name] = df
    print("[info] Loaded tables:", list(tables.keys()))
    return tables

def preprocess_los_cohort(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepare Length of Stay dataset from admissions."""
    admissions = tables["admissions"]
    admissions["los_days"] = (
        pd.to_datetime(admissions["dischtime"]) - pd.to_datetime(admissions["admittime"])
    ).dt.days
    return admissions[["subject_id", "hadm_id", "admittime", "dischtime", "los_days"]]

def preprocess_disease_risk(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepare Disease Risk dataset from admissions + diagnoses."""
    patients = tables["patients"]
    admissions = tables["admissions"]
    diagnoses = tables.get("diagnoses_icd")

    df = admissions.merge(patients, on="subject_id", how="left")

    if diagnoses is not None:
        diag_counts = diagnoses.groupby("hadm_id").size().reset_index(name="diag_count")
        df = df.merge(diag_counts, on="hadm_id", how="left").fillna({"diag_count": 0})
        df["high_risk"] = (df["diag_count"] > 5).astype(int)

    return df

def save_df(df: pd.DataFrame, name: str) -> None:
    out = RAW_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"[saved] {out}")

def main() -> None:
    print("[data] loading MIMIC-IV demo from local folder:", DATA_DIR)
    tables = load_tables()

    los_df = preprocess_los_cohort(tables)
    risk_df = preprocess_disease_risk(tables)

    save_df(los_df, "mimic_los_cohort")
    save_df(risk_df, "mimic_disease_risk")

    print("[done]")

if __name__ == "__main__":
    main()
