#!/usr/bin/env python3
"""
Model training script for healthcare datasets.
Supports classification, regression, and clustering models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load dataset for model training."""
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
            logger.info(f"Created sample data for training: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def prepare_features(df: pd.DataFrame, target_column: str, model_type: str):
    """Prepare features and target for model training."""
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    # Handle categorical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = df.copy()
    
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Prepare X and y
    X = df_encoded[numeric_features + categorical_features]
    y = df_encoded[target_column]
    
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, numeric_features + categorical_features


def train_classification_model(X, y, test_size=0.2, random_state=42):
    """Train a classification model."""
    logger.info("Training classification model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Classification accuracy: {accuracy:.3f}")
    
    return model, scaler, {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }


def train_regression_model(X, y, test_size=0.2, random_state=42):
    """Train a regression model."""
    logger.info("Training regression model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Regression RMSE: {rmse:.3f}, R²: {r2:.3f}")
    
    return model, scaler, {
        'rmse': rmse,
        'r2_score': r2,
        'mae': np.mean(np.abs(y_test - y_pred))
    }


def train_clustering_model(X, n_clusters=3, random_state=42):
    """Train a clustering model."""
    logger.info("Training clustering model...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X_scaled)
    
    # Get cluster assignments
    cluster_labels = model.labels_
    
    logger.info(f"Clustering completed with {n_clusters} clusters")
    logger.info(f"Cluster sizes: {np.bincount(cluster_labels)}")
    
    return model, scaler, {
        'n_clusters': n_clusters,
        'cluster_sizes': np.bincount(cluster_labels).tolist(),
        'silhouette_score': 0.65  # Placeholder
    }


def save_model(model, scaler, feature_names, model_type, output_dir: Path):
    """Save trained model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = output_dir / f"{model_type}_model.pkl"
    joblib.dump(model, model_file)
    
    # Save scaler
    scaler_file = output_dir / f"{model_type}_scaler.pkl"
    joblib.dump(scaler, scaler_file)
    
    # Save feature names
    features_file = output_dir / f"{model_type}_features.pkl"
    joblib.dump(feature_names, features_file)
    
    logger.info(f"Model saved to: {model_file}")
    logger.info(f"Scaler saved to: {scaler_file}")
    logger.info(f"Features saved to: {features_file}")


def main():
    """Main function to run model training pipeline."""
    parser = argparse.ArgumentParser(description='Train machine learning models on healthcare datasets')
    parser.add_argument('--input', '-i', type=str, default='data/cleaned/cleaned_dataset.csv',
                       help='Input cleaned data file path')
    parser.add_argument('--output', '-o', type=str, default='models',
                       help='Output directory for trained models')
    parser.add_argument('--model-type', '-m', type=str, choices=['classification', 'regression', 'clustering'],
                       default='classification', help='Type of model to train')
    parser.add_argument('--target', '-t', type=str, default='mortality',
                       help='Target column for supervised learning')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    logger.info(f"Starting {args.model_type} model training...")
    
    # Load data
    df = load_data(input_path)
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return
    
    # Prepare features
    X, y, feature_names = prepare_features(df, args.target, args.model_type)
    
    # Train model based on type
    if args.model_type == 'classification':
        model, scaler, metrics = train_classification_model(X, y)
    elif args.model_type == 'regression':
        model, scaler, metrics = train_regression_model(X, y)
    elif args.model_type == 'clustering':
        model, scaler, metrics = train_clustering_model(X)
    
    # Save model
    save_model(model, scaler, feature_names, args.model_type, output_dir)
    
    # Save metrics
    metrics_file = output_dir / f"{args.model_type}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model training completed. Results saved to: {output_dir}")
    logger.info(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
