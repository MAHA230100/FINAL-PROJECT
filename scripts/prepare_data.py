#!/usr/bin/env python3
"""
Data preparation script for healthcare dataset.
Cleans and prepares the healthcare_ai_dataset_500_patients.csv for analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_clean_data(input_path: Path) -> pd.DataFrame:
    """Load and clean the healthcare dataset."""
    logger.info(f"Loading data from {input_path}")
    
    # Load the dataset
    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Basic info
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    
    # Clean the data
    initial_rows = len(df)
    
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df_cleaned)
    logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = df_cleaned.isnull().sum().sum()
    
    # For categorical columns, fill with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    # For numerical columns, fill with median
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
    
    missing_after = df_cleaned.isnull().sum().sum()
    logger.info(f"Handled {missing_before - missing_after} missing values")
    
    # Create additional features
    logger.info("Creating additional features...")
    
    # Age groups
    df_cleaned['age_group'] = pd.cut(df_cleaned['age'], 
                                   bins=[0, 30, 50, 70, 100], 
                                   labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # Length of stay categories
    df_cleaned['los_category'] = pd.cut(df_cleaned['length_of_stay'], 
                                      bins=[0, 3, 7, 14, 100], 
                                      labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    # Risk score based on multiple factors
    df_cleaned['risk_score'] = (
        (df_cleaned['age'] / 100) * 0.3 +
        (df_cleaned['length_of_stay'] / 20) * 0.3 +
        (df_cleaned['previous_admissions'] / 5) * 0.2 +
        (df_cleaned['vitals_bp'] / 200) * 0.2
    )
    
    logger.info(f"Final dataset shape: {df_cleaned.shape}")
    return df_cleaned


def prepare_features_for_ml(df: pd.DataFrame) -> tuple:
    """Prepare features for machine learning models."""
    logger.info("Preparing features for machine learning...")
    
    # Select features for ML
    feature_cols = [
        'age', 'gender', 'admission_type', 'admission_location', 
        'insurance', 'language', 'marital_status', 'drg_type',
        'comorbidities', 'lab_results', 'vitals_bp', 'vitals_hr', 
        'previous_admissions', 'risk_score'
    ]
    
    # Create feature matrix
    X = df[feature_cols].copy()
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Create target variables
    # Classification target: outcome (Recovered, Readmitted, Deceased)
    y_classification = df['outcome'].copy()
    
    # Regression target: length_of_stay
    y_regression = df['length_of_stay'].copy()
    
    # Clustering features (normalized)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Classification target distribution:\n{y_classification.value_counts()}")
    logger.info(f"Regression target stats: mean={y_regression.mean():.2f}, std={y_regression.std():.2f}")
    
    return X, y_classification, y_regression, X_scaled, label_encoders, scaler


def save_processed_data(df: pd.DataFrame, X: pd.DataFrame, y_classification: pd.Series, 
                       y_regression: pd.Series, X_scaled: np.ndarray, 
                       label_encoders: dict, scaler: StandardScaler, output_dir: Path):
    """Save processed data and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned dataset
    cleaned_file = output_dir / 'cleaned_healthcare_dataset.csv'
    df.to_csv(cleaned_file, index=False)
    logger.info(f"Saved cleaned dataset to: {cleaned_file}")
    
    # Save feature matrix
    features_file = output_dir / 'features.csv'
    X.to_csv(features_file, index=False)
    logger.info(f"Saved features to: {features_file}")
    
    # Save targets
    targets_file = output_dir / 'targets.csv'
    targets_df = pd.DataFrame({
        'classification_target': y_classification,
        'regression_target': y_regression
    })
    targets_df.to_csv(targets_file, index=False)
    logger.info(f"Saved targets to: {targets_file}")
    
    # Save scaled features for clustering
    scaled_file = output_dir / 'features_scaled.csv'
    pd.DataFrame(X_scaled, columns=X.columns).to_csv(scaled_file, index=False)
    logger.info(f"Saved scaled features to: {scaled_file}")
    
    # Save metadata
    import joblib
    metadata = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_columns': list(X.columns),
        'target_columns': ['classification_target', 'regression_target']
    }
    
    metadata_file = output_dir / 'metadata.pkl'
    joblib.dump(metadata, metadata_file)
    logger.info(f"Saved metadata to: {metadata_file}")
    
    # Save summary statistics
    summary = {
        'dataset_shape': df.shape,
        'feature_shape': X.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'categorical_summary': {},
        'numerical_summary': df.describe().to_dict()
    }
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    import json
    summary_file = output_dir / 'data_summary.json'
    with open(summary_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_summary = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                json_summary[key] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in value.items()}
            else:
                json_summary[key] = value.tolist() if hasattr(value, 'tolist') else value
        json.dump(json_summary, f, indent=2)
    
    logger.info(f"Saved data summary to: {summary_file}")


def main():
    """Main function to run data preparation pipeline."""
    parser = argparse.ArgumentParser(description='Prepare healthcare dataset for analysis')
    parser.add_argument('--input', '-i', type=str, default='data/raw/healthcare_ai_dataset_500_patients.csv',
                       help='Input data file path')
    parser.add_argument('--output', '-o', type=str, default='data/cleaned',
                       help='Output directory for processed data')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    logger.info("Starting data preparation pipeline...")
    
    # Load and clean data
    df_cleaned = load_and_clean_data(input_path)
    
    # Prepare features for ML
    X, y_classification, y_regression, X_scaled, label_encoders, scaler = prepare_features_for_ml(df_cleaned)
    
    # Save processed data
    save_processed_data(df_cleaned, X, y_classification, y_regression, X_scaled, 
                       label_encoders, scaler, output_dir)
    
    logger.info("Data preparation completed successfully!")
    logger.info(f"Processed data saved to: {output_dir}")


if __name__ == "__main__":
    main()
