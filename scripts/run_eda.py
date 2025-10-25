#!/usr/bin/env python3
"""
EDA script for healthcare datasets.
Replicates the logic from the eda.ipynb notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cleaned_data(data_path: Path) -> pd.DataFrame:
    """Load cleaned dataset for EDA."""
    try:
        if data_path.exists():
            df = pd.read_csv(data_path)
            logger.info(f"Loaded cleaned data from {data_path}: {df.shape}")
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
            logger.info(f"Created sample data for EDA: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset."""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if not df.empty else {},
        'categorical_summary': {}
    }
    
    # Categorical variables summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    logger.info("Dataset Summary:")
    logger.info(f"Shape: {summary['shape']}")
    logger.info(f"Columns: {summary['columns']}")
    logger.info(f"Missing values: {summary['missing_values']}")
    
    return summary


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create EDA visualizations and save them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Age distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Length of stay distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['length_of_stay'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Length of Stay Distribution')
    plt.xlabel('Length of Stay (days)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'los_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gender distribution
    plt.figure(figsize=(8, 6))
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Admission type distribution
    plt.figure(figsize=(10, 6))
    admission_counts = df['admission_type'].value_counts()
    plt.bar(admission_counts.index, admission_counts.values)
    plt.title('Admission Type Distribution')
    plt.xlabel('Admission Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'admission_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation heatmap (numeric variables only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Age vs Length of Stay scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['age'], df['length_of_stay'], alpha=0.6, color='green')
    plt.title('Age vs Length of Stay')
    plt.xlabel('Age')
    plt.ylabel('Length of Stay (days)')
    plt.tight_layout()
    plt.savefig(output_dir / 'age_vs_los.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualizations to: {output_dir}")


def main():
    """Main function to run EDA pipeline."""
    parser = argparse.ArgumentParser(description='Run EDA on healthcare datasets')
    parser.add_argument('--input', '-i', type=str, default='data/cleaned/cleaned_dataset.csv',
                       help='Input cleaned data file path')
    parser.add_argument('--output', '-o', type=str, default='data/eda_results',
                       help='Output directory for EDA results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    logger.info("Starting EDA pipeline...")
    
    # Load data
    df = load_cleaned_data(input_path)
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return
    
    # Generate summary statistics
    summary = generate_summary_statistics(df)
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Save summary statistics
    summary_file = output_dir / 'summary_statistics.json'
    import json
    with open(summary_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_summary = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                json_summary[key] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in value.items()}
            else:
                json_summary[key] = value.tolist() if hasattr(value, 'tolist') else value
        json.dump(json_summary, f, indent=2)
    
    logger.info(f"EDA completed. Results saved to: {output_dir}")
    logger.info(f"Summary statistics saved to: {summary_file}")


if __name__ == "__main__":
    main()
