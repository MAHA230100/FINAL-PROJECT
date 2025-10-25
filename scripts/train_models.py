#!/usr/bin/env python3
"""
Comprehensive model training script for healthcare dataset.
Trains classification, regression, and clustering models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, adjusted_rand_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data(input_path: Path) -> tuple:
    """Load and prepare the healthcare dataset for training."""
    logger.info(f"Loading data from {input_path}")
    
    # Load the dataset
    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Clean the data
    df_cleaned = df.drop_duplicates()
    logger.info(f"After removing duplicates: {df_cleaned.shape}")
    
    # Handle missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # Create additional features
    df_cleaned['age_group'] = pd.cut(df_cleaned['age'], 
                                   bins=[0, 30, 50, 70, 100], 
                                   labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    df_cleaned['los_category'] = pd.cut(df_cleaned['length_of_stay'], 
                                      bins=[0, 3, 7, 14, 100], 
                                      labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    # Risk score
    df_cleaned['risk_score'] = (
        (df_cleaned['age'] / 100) * 0.3 +
        (df_cleaned['length_of_stay'] / 20) * 0.3 +
        (df_cleaned['previous_admissions'] / 5) * 0.2 +
        (df_cleaned['vitals_bp'] / 200) * 0.2
    )
    
    return df_cleaned


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features for machine learning."""
    logger.info("Preparing features for ML...")
    
    # Select features
    feature_cols = [
        'age', 'gender', 'admission_type', 'admission_location', 
        'insurance', 'language', 'marital_status', 'drg_type',
        'comorbidities', 'lab_results', 'vitals_bp', 'vitals_hr', 
        'previous_admissions', 'risk_score'
    ]
    
    X = df[feature_cols].copy()
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Create targets
    y_classification = df['outcome'].copy()
    y_regression = df['length_of_stay'].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Classification target distribution:\n{y_classification.value_counts()}")
    logger.info(f"Regression target stats: mean={y_regression.mean():.2f}, std={y_regression.std():.2f}")
    
    return X, y_classification, y_regression, X_scaled, label_encoders, scaler


def train_classification_models(X: np.ndarray, y: pd.Series, output_dir: Path) -> dict:
    """Train classification models for outcome prediction."""
    logger.info("Training classification models...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                metrics['auc'] = auc
            except:
                metrics['auc'] = None
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'test_labels': y_test
        }
        
        logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Save best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['f1_score'])
    best_model = results[best_model_name]['model']
    
    model_path = output_dir / 'classification' / 'best_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    
    # Save all results
    results_path = output_dir / 'classification' / 'results.json'
    results_json = {}
    for name, result in results.items():
        results_json[name] = {
            'metrics': result['metrics'],
            'model_type': type(result['model']).__name__
        }
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Best classification model: {best_model_name}")
    logger.info(f"Results saved to: {results_path}")
    
    return results


def train_regression_models(X: np.ndarray, y: pd.Series, output_dir: Path) -> dict:
    """Train regression models for length of stay prediction."""
    logger.info("Training regression models...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'SVR': SVR(kernel='rbf')
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'test_labels': y_test
        }
        
        logger.info(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Save best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['r2_score'])
    best_model = results[best_model_name]['model']
    
    model_path = output_dir / 'regression' / 'best_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    
    # Save all results
    results_path = output_dir / 'regression' / 'results.json'
    results_json = {}
    for name, result in results.items():
        results_json[name] = {
            'metrics': result['metrics'],
            'model_type': type(result['model']).__name__
        }
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Best regression model: {best_model_name}")
    logger.info(f"Results saved to: {results_path}")
    
    return results


def train_clustering_models(X_scaled: np.ndarray, output_dir: Path) -> dict:
    """Train clustering models for patient segmentation."""
    logger.info("Training clustering models...")
    
    models = {
        'KMeans': KMeans(n_clusters=4, random_state=42, n_init=10),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Fit model
        clusters = model.fit_predict(X_scaled)
        
        # Metrics
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_samples': len(clusters)
        }
        
        if n_clusters > 1:
            try:
                silhouette = silhouette_score(X_scaled, clusters)
                metrics['silhouette_score'] = silhouette
            except:
                metrics['silhouette_score'] = None
        
        results[name] = {
            'model': model,
            'clusters': clusters,
            'metrics': metrics
        }
        
        logger.info(f"{name} - Clusters: {n_clusters}, Noise: {n_noise}")
    
    # Save best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['metrics'].get('silhouette_score', 0))
    best_model = results[best_model_name]['model']
    
    model_path = output_dir / 'clustering' / 'best_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    
    # Save all results
    results_path = output_dir / 'clustering' / 'results.json'
    results_json = {}
    for name, result in results.items():
        results_json[name] = {
            'metrics': result['metrics'],
            'model_type': type(result['model']).__name__
        }
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Best clustering model: {best_model_name}")
    logger.info(f"Results saved to: {results_path}")
    
    return results


def create_model_visualizations(results: dict, output_dir: Path):
    """Create visualizations for model results."""
    logger.info("Creating model visualizations...")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Classification results
    if 'classification' in str(output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Classification Model Performance', fontsize=16)
        
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [results[model]['metrics'][metric] for model in models]
            ax.bar(models, values)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'classification_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Regression results
    if 'regression' in str(output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Regression Model Performance', fontsize=16)
        
        models = list(results.keys())
        metrics = ['mse', 'rmse', 'mae', 'r2_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [results[model]['metrics'][metric] for model in models]
            ax.bar(models, values)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'regression_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizations saved to: {viz_dir}")


def main():
    """Main function to run model training pipeline."""
    parser = argparse.ArgumentParser(description='Train ML models on healthcare dataset')
    parser.add_argument('--input', '-i', type=str, default='data/raw/healthcare_ai_dataset_500_patients.csv',
                       help='Input data file path')
    parser.add_argument('--output', '-o', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    logger.info("Starting model training pipeline...")
    
    # Load and prepare data
    df = load_and_prepare_data(input_path)
    X, y_classification, y_regression, X_scaled, label_encoders, scaler = prepare_features(df)
    
    # Train models
    logger.info("=" * 50)
    logger.info("TRAINING CLASSIFICATION MODELS")
    logger.info("=" * 50)
    classification_results = train_classification_models(X, y_classification, output_dir)
    
    logger.info("=" * 50)
    logger.info("TRAINING REGRESSION MODELS")
    logger.info("=" * 50)
    regression_results = train_regression_models(X, y_regression, output_dir)
    
    logger.info("=" * 50)
    logger.info("TRAINING CLUSTERING MODELS")
    logger.info("=" * 50)
    clustering_results = train_clustering_models(X_scaled, output_dir)
    
    # Create visualizations
    create_model_visualizations(classification_results, output_dir / 'classification')
    create_model_visualizations(regression_results, output_dir / 'regression')
    
    # Save metadata
    metadata = {
        'feature_columns': list(X.columns),
        'label_encoders': {col: le.classes_.tolist() for col, le in label_encoders.items()},
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'dataset_info': {
            'n_samples': len(df),
            'n_features': X.shape[1],
            'classification_classes': y_classification.unique().tolist(),
            'regression_range': [float(y_regression.min()), float(y_regression.max())]
        }
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Model training pipeline completed successfully!")
    logger.info(f"Models and results saved to: {output_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
