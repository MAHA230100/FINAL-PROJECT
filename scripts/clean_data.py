#!/usr/bin/env python3
"""
Data cleaning script for healthcare datasets.
Replicates the logic from the data_cleaning.ipynb notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load dataset from file or create sample data."""
    try:
        if data_path.exists():
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path}: {df.shape}")
        else:
            # Create sample data for demonstration
            np.random.seed(42)
            df = pd.DataFrame({
                'patient_id': range(1000),
                'age': np.random.normal(65, 15, 1000),
                'gender': np.random.choice(['M', 'F'], 1000),
                'admission_type': np.random.choice(['URGENT', 'ELECTIVE', 'EMERGENCY'], 1000),
                'length_of_stay': np.random.exponential(5, 1000),
                'mortality': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
            })
            logger.info(f"Created sample data: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by removing duplicates and missing values."""
    initial_rows = len(df)
    
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df_cleaned)
    logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = df_cleaned.isnull().sum().sum()
    df_cleaned = df_cleaned.dropna()
    missing_after = df_cleaned.isnull().sum().sum()
    logger.info(f"Removed {missing_before - missing_after} rows with missing values")
    
    logger.info(f"Final dataset shape: {df_cleaned.shape}")
    return df_cleaned


def validate_data(df: pd.DataFrame) -> dict:
    """Validate data quality and return summary statistics."""
    validation_results = {
        'age_range': (df['age'].min(), df['age'].max()),
        'los_range': (df['length_of_stay'].min(), df['length_of_stay'].max()),
        'mortality_rate': df['mortality'].mean(),
        'total_rows': len(df)
    }
    
    logger.info("Data Quality Checks:")
    logger.info(f"Age range: {validation_results['age_range'][0]:.1f} - {validation_results['age_range'][1]:.1f}")
    logger.info(f"Length of stay range: {validation_results['los_range'][0]:.1f} - {validation_results['los_range'][1]:.1f}")
    logger.info(f"Mortality rate: {validation_results['mortality_rate']:.3f}")
    
    return validation_results


def main():
    """Main function to run data cleaning pipeline."""
    parser = argparse.ArgumentParser(description='Clean healthcare datasets')
    parser.add_argument('--input', '-i', type=str, default='data/raw/sample_data.csv',
                       help='Input data file path')
    parser.add_argument('--output', '-o', type=str, default='data/cleaned/cleaned_dataset.csv',
                       help='Output cleaned data file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting data cleaning pipeline...")
    
    # Load data
    df = load_data(input_path)
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Validate data
    validation_results = validate_data(df_cleaned)
    
    # Save cleaned data
    df_cleaned.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to: {output_path}")
    
    # Print summary
    initial_rows = len(df)
    final_rows = len(df_cleaned)
    reduction_pct = ((initial_rows - final_rows) / initial_rows * 100) if initial_rows > 0 else 0
    
    logger.info("Data Cleaning Summary:")
    logger.info(f"Original rows: {initial_rows}")
    logger.info(f"Final rows: {final_rows}")
    logger.info(f"Data reduction: {reduction_pct:.1f}%")
    logger.info(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    main()
