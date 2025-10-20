#!/usr/bin/env python3
"""
Process cleaned MIMIC-IV data for model training and evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib

CLEAN_DIR = Path("healthai/data/processed/mimic")
MODELS_DIR = Path("healthai/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_mortality_data():
    """Prepare mortality prediction dataset"""
    print("Preparing mortality prediction dataset...")
    
    df = pd.read_csv(CLEAN_DIR / "mortality_prediction.csv")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'mortality']]
    X = df[feature_cols]
    y = df['mortality']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def prepare_los_data():
    """Prepare length of stay prediction dataset"""
    print("Preparing length of stay prediction dataset...")
    
    df = pd.read_csv(CLEAN_DIR / "los_prediction.csv")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'los_days']]
    X = df[feature_cols]
    y = df['los_days']
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def train_mortality_model():
    """Train mortality prediction model"""
    print("\nTraining mortality prediction model...")
    
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_mortality_data()
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nMortality Prediction Results:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "mimic_mortality_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "mimic_mortality_scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "mimic_mortality_features.pkl")
    
    print(f"Model saved to: {MODELS_DIR / 'mimic_mortality_model.pkl'}")
    
    return model, scaler, feature_cols

def train_los_model():
    """Train length of stay prediction model"""
    print("\nTraining length of stay prediction model...")
    
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_los_data()
    
    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\nLength of Stay Prediction Results:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.3f}")
    
    # Save model and scaler
    joblib.dump(model, MODELS_DIR / "mimic_los_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "mimic_los_scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "mimic_los_features.pkl")
    
    print(f"Model saved to: {MODELS_DIR / 'mimic_los_model.pkl'}")
    
    return model, scaler, feature_cols

def create_feature_importance_report():
    """Create feature importance reports"""
    print("\nCreating feature importance reports...")
    
    # Load models and features
    mortality_model = joblib.load(MODELS_DIR / "mimic_mortality_model.pkl")
    mortality_features = joblib.load(MODELS_DIR / "mimic_mortality_features.pkl")
    
    los_model = joblib.load(MODELS_DIR / "mimic_los_model.pkl")
    los_features = joblib.load(MODELS_DIR / "mimic_los_features.pkl")
    
    # Mortality feature importance
    mortality_importance = pd.DataFrame({
        'feature': mortality_features,
        'importance': mortality_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features for Mortality Prediction:")
    print(mortality_importance.head(10))
    
    # LOS feature importance
    los_importance = pd.DataFrame({
        'feature': los_features,
        'importance': los_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features for Length of Stay Prediction:")
    print(los_importance.head(10))
    
    # Save importance reports
    mortality_importance.to_csv(CLEAN_DIR / "mortality_feature_importance.csv", index=False)
    los_importance.to_csv(CLEAN_DIR / "los_feature_importance.csv", index=False)

def main():
    print("Processing MIMIC-IV data for machine learning...")
    
    # Train models
    mortality_model, mortality_scaler, mortality_features = train_mortality_model()
    los_model, los_scaler, los_features = train_los_model()
    
    # Create feature importance reports
    create_feature_importance_report()
    
    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Reports saved to: {CLEAN_DIR}")
    print("\nAvailable models:")
    print("- mimic_mortality_model.pkl (classification)")
    print("- mimic_los_model.pkl (regression)")

if __name__ == "__main__":
    main()
