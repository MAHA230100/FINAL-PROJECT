#!/usr/bin/env python3
"""
Extract and decompress MIMIC-IV demo data files.
"""

import gzip
import shutil
from pathlib import Path
import pandas as pd

MIMIC_ROOT = Path("../../scripts/data/mimic_demo/mimic-iv-clinical-database-demo-2.2")
OUTPUT_DIR = Path("healthai/data/raw/mimic")

def extract_gz_files():
    """Extract all .csv.gz files to healthai/data/raw/mimic/"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    # Extract hosp files
    hosp_dir = MIMIC_ROOT / "hosp"
    for gz_file in hosp_dir.glob("*.csv.gz"):
        output_file = OUTPUT_DIR / f"hosp_{gz_file.stem.replace('.csv', '')}.csv"
        
        print(f"Extracting {gz_file.name} -> {output_file.name}")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        extracted_count += 1
    
    # Extract icu files
    icu_dir = MIMIC_ROOT / "icu"
    for gz_file in icu_dir.glob("*.csv.gz"):
        output_file = OUTPUT_DIR / f"icu_{gz_file.stem.replace('.csv', '')}.csv"
        
        print(f"Extracting {gz_file.name} -> {output_file.name}")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        extracted_count += 1
    
    # Copy demo subject IDs
    demo_subjects = MIMIC_ROOT / "demo_subject_id.csv"
    if demo_subjects.exists():
        shutil.copy2(demo_subjects, OUTPUT_DIR / "demo_subject_ids.csv")
        print(f"Copied {demo_subjects.name}")
    
    print(f"\nExtracted {extracted_count} files to {OUTPUT_DIR}")
    return OUTPUT_DIR

def show_data_summary():
    """Show summary of extracted data files"""
    print("\n" + "="*50)
    print("MIMIC-IV Demo Data Summary")
    print("="*50)
    
    for csv_file in sorted(OUTPUT_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            print(f"{csv_file.name:30} | {df.shape[0]:6,} rows | {df.shape[1]:3} cols")
        except Exception as e:
            print(f"{csv_file.name:30} | ERROR: {e}")

if __name__ == "__main__":
    extract_gz_files()
    show_data_summary()
