#!/usr/bin/env python3
"""
Clean and preprocess MIMIC-IV demo data for ML tasks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = Path("healthai/data/raw/mimic")
CLEAN_DIR = Path("healthai/data/processed/mimic")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def clean_patients():
    """Clean patients table"""
    print("Cleaning patients table...")
    df = pd.read_csv(RAW_DIR / "hosp_patients.csv")
    
    # Convert dates
    df['dod'] = pd.to_datetime(df['dod'], errors='coerce')
    df['anchor_age'] = pd.to_numeric(df['anchor_age'], errors='coerce')
    
    # Handle missing values
    df['anchor_age'] = df['anchor_age'].fillna(df['anchor_age'].median())
    
    # Create binary mortality flag (if death date exists)
    df['mortality'] = df['dod'].notna().astype(int)
    
    # Clean gender
    df['gender'] = df['gender'].str.upper()
    df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(-1)
    
    return df

def clean_admissions():
    """Clean admissions table"""
    print("Cleaning admissions table...")
    df = pd.read_csv(RAW_DIR / "hosp_admissions.csv")
    
    # Convert dates
    date_cols = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate length of stay
    df['los_days'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / (24*3600)
    df['los_days'] = df['los_days'].fillna(df['los_days'].median())
    
    # Clean admission type
    df['admission_type'] = df['admission_type'].fillna('UNKNOWN')
    df['admission_location'] = df['admission_location'].fillna('UNKNOWN')
    
    return df

def clean_labevents():
    """Clean lab events table"""
    print("Cleaning lab events table...")
    df = pd.read_csv(RAW_DIR / "hosp_labevents.csv")
    
    # Convert numeric values
    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    
    # Remove extreme outliers (beyond 3 standard deviations)
    numeric_cols = ['valuenum']
    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            df = df[(df[col] >= mean_val - 3*std_val) & 
                   (df[col] <= mean_val + 3*std_val)]
    
    # Convert charttime
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
    
    return df

def clean_chartevents():
    """Clean ICU chart events table"""
    print("Cleaning ICU chart events table...")
    df = pd.read_csv(RAW_DIR / "icu_chartevents.csv")
    
    # Convert numeric values
    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    
    # Convert datetime
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
    df['storetime'] = pd.to_datetime(df['storetime'], errors='coerce')
    
    # Remove extreme outliers
    numeric_cols = ['valuenum']
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df = df[(df[col] >= mean_val - 3*std_val) | 
                       (df[col] <= mean_val + 3*std_val) |
                       (df[col].isna())]
    
    return df

def create_ml_datasets():
    """Create datasets for ML tasks"""
    print("\nCreating ML datasets...")
    
    # Load cleaned tables
    patients = clean_patients()
    admissions = clean_admissions()
    labevents = clean_labevents()
    
    # 1. Mortality Prediction Dataset
    print("Creating mortality prediction dataset...")
    mortality_df = patients.merge(admissions, on='subject_id', how='left')
    
    # Select features for mortality prediction
    mortality_features = mortality_df[[
        'subject_id', 'anchor_age', 'gender', 'los_days', 
        'admission_type', 'admission_location', 'mortality'
    ]].copy()
    
    # Encode categorical variables
    mortality_features = pd.get_dummies(
        mortality_features, 
        columns=['admission_type', 'admission_location'],
        prefix=['adm_type', 'adm_loc']
    )
    
    mortality_features.to_csv(CLEAN_DIR / "mortality_prediction.csv", index=False)
    print(f"Saved mortality dataset: {mortality_features.shape}")
    
    # 2. Length of Stay Prediction Dataset
    print("Creating length of stay prediction dataset...")
    los_features = mortality_df[[
        'subject_id', 'anchor_age', 'gender', 'admission_type', 
        'admission_location', 'los_days'
    ]].copy()
    
    # Encode categorical variables
    los_features = pd.get_dummies(
        los_features, 
        columns=['admission_type', 'admission_location'],
        prefix=['adm_type', 'adm_loc']
    )
    
    # Remove rows with missing LOS
    los_features = los_features.dropna(subset=['los_days'])
    
    los_features.to_csv(CLEAN_DIR / "los_prediction.csv", index=False)
    print(f"Saved LOS dataset: {los_features.shape}")
    
    # 3. Lab Values Summary
    print("Creating lab values summary...")
    lab_summary = labevents.groupby(['subject_id']).agg({
        'valuenum': ['mean', 'std', 'min', 'max', 'count']
    }).round(3)
    
    # Flatten column names
    lab_summary.columns = [f"lab_{col[1]}" for col in lab_summary.columns]
    lab_summary = lab_summary.reset_index()
    
    lab_summary.to_csv(CLEAN_DIR / "lab_summary.csv", index=False)
    print(f"Saved lab summary: {lab_summary.shape}")
    
    return mortality_features, los_features, lab_summary

def main():
    print("Starting MIMIC-IV data cleaning...")
    
    # Clean individual tables
    mortality_df, los_df, lab_df = create_ml_datasets()
    
    print("\n" + "="*50)
    print("Data Cleaning Summary")
    print("="*50)
    print(f"Mortality prediction dataset: {mortality_df.shape}")
    print(f"Length of stay dataset: {los_df.shape}")
    print(f"Lab values summary: {lab_df.shape}")
    print(f"\nCleaned files saved to: {CLEAN_DIR}")
    
    # Show sample statistics
    print("\nMortality rates:")
    print(mortality_df['mortality'].value_counts(normalize=True))
    
    print("\nLOS statistics (days):")
    print(los_df['los_days'].describe())

if __name__ == "__main__":
    main()
