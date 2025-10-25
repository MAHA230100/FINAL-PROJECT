"""
Complete HealthAI Pipeline - Run the entire data science pipeline
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_health_dataset():
    """Load the health AI dataset"""
    try:
        # Try to load from the data/raw directory
        dataset_path = "data/raw/health_ai_dataset_500_patients.csv"
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        else:
            print(f"❌ Dataset not found at {dataset_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def clean_health_data(df):
    """Clean the health dataset"""
    print("🧹 Cleaning health data...")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    # Fill missing values with appropriate methods
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            # Numerical columns - fill with median
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            # Categorical columns - fill with mode
            mode_value = df_clean[col].mode()
            if not mode_value.empty:
                df_clean[col].fillna(mode_value[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    
    # Remove duplicates
    duplicates_before = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    duplicates_after = df_clean.duplicated().sum()
    print(f"Duplicates removed: {duplicates_before - duplicates_after}")
    
    # Data type optimization
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to category if it has few unique values
            if df_clean[col].nunique() < 20:
                df_clean[col] = df_clean[col].astype('category')
    
    print(f"✅ Data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean

def perform_eda(df):
    """Perform exploratory data analysis"""
    print("📊 Performing EDA...")
    
    eda_results = {
        'dataset_info': {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        },
        'basic_stats': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'correlation_matrix': df.corr(numeric_only=True).to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
    }
    
    # Save EDA results
    os.makedirs("data/eda_results", exist_ok=True)
    with open("data/eda_results/health_dataset_eda.json", "w") as f:
        json.dump(eda_results, f, indent=4, default=str)
    
    print("✅ EDA completed and saved")
    return eda_results

def train_models(df):
    """Train machine learning models"""
    print("🤖 Training models...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    # Prepare features and targets
    # For classification: predict a categorical outcome
    # For regression: predict a numerical outcome
    
    # Classification model - predict if patient has high risk
    if 'risk_score' in df.columns:
        # Use risk_score as target for classification
        X_class = df.drop(['risk_score'], axis=1)
        y_class = (df['risk_score'] > df['risk_score'].median()).astype(int)
        
        # Handle categorical variables
        X_class = pd.get_dummies(X_class, drop_first=True)
        
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42
        )
        
        # Train classification model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_class, y_train_class)
        
        # Evaluate
        y_pred_class = clf.predict(X_test_class)
        classification_metrics = {
            'accuracy': accuracy_score(y_test_class, y_pred_class),
            'precision': precision_score(y_test_class, y_pred_class, average='weighted'),
            'recall': recall_score(y_test_class, y_pred_class, average='weighted'),
            'f1_score': f1_score(y_test_class, y_pred_class, average='weighted')
        }
        
        # Save model
        os.makedirs("models/classification", exist_ok=True)
        joblib.dump(clf, "models/classification/health_risk_classifier.joblib")
        
        print(f"✅ Classification model trained. Accuracy: {classification_metrics['accuracy']:.3f}")
    else:
        print("⚠️ No suitable target for classification found")
        classification_metrics = None
    
    # Regression model - predict a numerical outcome
    if 'age' in df.columns and 'length_of_stay' in df.columns:
        # Use age and other features to predict length of stay
        X_reg = df.drop(['length_of_stay'], axis=1)
        y_reg = df['length_of_stay']
        
        # Handle categorical variables
        X_reg = pd.get_dummies(X_reg, drop_first=True)
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Train regression model
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train_reg, y_train_reg)
        
        # Evaluate
        y_pred_reg = reg.predict(X_test_reg)
        regression_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
            'mae': np.mean(np.abs(y_test_reg - y_pred_reg)),
            'r2_score': r2_score(y_test_reg, y_pred_reg)
        }
        
        # Save model
        os.makedirs("models/regression", exist_ok=True)
        joblib.dump(reg, "models/regression/length_of_stay_predictor.joblib")
        
        print(f"✅ Regression model trained. R²: {regression_metrics['r2_score']:.3f}")
    else:
        print("⚠️ No suitable target for regression found")
        regression_metrics = None
    
    # Save model metrics
    model_metrics = {
        'classification': classification_metrics,
        'regression': regression_metrics,
        'training_date': datetime.now().isoformat()
    }
    
    with open("models/model_metrics.json", "w") as f:
        json.dump(model_metrics, f, indent=4)
    
    return model_metrics

def generate_insights(df, eda_results, model_metrics):
    """Generate insights and recommendations"""
    print("💡 Generating insights...")
    
    insights = {
        'dataset_summary': {
            'total_patients': len(df),
            'total_features': len(df.columns),
            'data_quality': 'Good' if df.isnull().sum().sum() == 0 else 'Needs attention'
        },
        'key_findings': [],
        'recommendations': [],
        'model_performance': model_metrics
    }
    
    # Key findings
    if 'age' in df.columns:
        avg_age = df['age'].mean()
        insights['key_findings'].append(f"Average patient age: {avg_age:.1f} years")
    
    if 'length_of_stay' in df.columns:
        avg_los = df['length_of_stay'].mean()
        insights['key_findings'].append(f"Average length of stay: {avg_los:.1f} days")
    
    if 'risk_score' in df.columns:
        high_risk_patients = (df['risk_score'] > df['risk_score'].median()).sum()
        insights['key_findings'].append(f"High-risk patients: {high_risk_patients} ({high_risk_patients/len(df)*100:.1f}%)")
    
    # Recommendations
    insights['recommendations'] = [
        "Implement risk stratification protocols",
        "Optimize length of stay management",
        "Enhance patient monitoring systems",
        "Develop predictive models for early intervention"
    ]
    
    # Save insights
    with open("data/insights/health_insights.json", "w") as f:
        json.dump(insights, f, indent=4, default=str)
    
    print("✅ Insights generated and saved")
    return insights

def main():
    """Run the complete HealthAI pipeline"""
    print("🚀 Starting HealthAI Complete Pipeline")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n📁 Step 1: Loading Health Dataset")
    df = load_health_dataset()
    if df is None:
        print("❌ Cannot proceed without dataset")
        return
    
    # Step 2: Clean data
    print("\n🧹 Step 2: Cleaning Data")
    df_clean = clean_health_data(df)
    
    # Save cleaned data
    os.makedirs("data/cleaned", exist_ok=True)
    df_clean.to_csv("data/cleaned/health_ai_dataset_500_patients_cleaned.csv", index=False)
    print("✅ Cleaned data saved")
    
    # Step 3: Perform EDA
    print("\n📊 Step 3: Exploratory Data Analysis")
    eda_results = perform_eda(df_clean)
    
    # Step 4: Train models
    print("\n🤖 Step 4: Training Models")
    model_metrics = train_models(df_clean)
    
    # Step 5: Generate insights
    print("\n💡 Step 5: Generating Insights")
    insights = generate_insights(df_clean, eda_results, model_metrics)
    
    # Step 6: Summary
    print("\n📋 Step 6: Pipeline Summary")
    print("=" * 50)
    print(f"✅ Dataset loaded: {df.shape[0]} patients, {df.shape[1]} features")
    print(f"✅ Data cleaned: {df_clean.shape[0]} patients, {df_clean.shape[1]} features")
    print(f"✅ EDA completed: Results saved to data/eda_results/")
    print(f"✅ Models trained: Metrics saved to models/")
    print(f"✅ Insights generated: Saved to data/insights/")
    
    print("\n🎉 HealthAI Pipeline completed successfully!")
    print("📁 Check the following directories for results:")
    print("   - data/cleaned/ - Cleaned dataset")
    print("   - data/eda_results/ - EDA results")
    print("   - data/insights/ - Generated insights")
    print("   - models/ - Trained models")

if __name__ == "__main__":
    main()