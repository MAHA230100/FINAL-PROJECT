"""
Comprehensive EDA for HealthAI Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_health_dataset():
    """Load the health AI dataset"""
    try:
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

def perform_comprehensive_eda(df):
    """Perform comprehensive exploratory data analysis"""
    print("📊 Performing Comprehensive EDA...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs("data/eda_results/health_dataset", exist_ok=True)
    os.makedirs("data/eda_results/health_dataset/plots", exist_ok=True)
    
    # 1. Dataset Overview
    print("📋 1. Dataset Overview")
    dataset_info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # 2. Basic Statistics
    print("📈 2. Basic Statistics")
    basic_stats = df.describe(include='all').to_dict()
    
    # 3. Data Quality Assessment
    print("🔍 3. Data Quality Assessment")
    data_quality = {
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_percentage': (df.duplicated().sum() / len(df) * 100),
        'unique_values_per_column': {col: df[col].nunique() for col in df.columns},
        'data_types': df.dtypes.to_dict()
    }
    
    # 4. Categorical Analysis
    print("📊 4. Categorical Analysis")
    categorical_analysis = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        categorical_analysis[col] = {
            'value_counts': df[col].value_counts().to_dict(),
            'unique_count': df[col].nunique(),
            'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None
        }
    
    # 5. Numerical Analysis
    print("📈 5. Numerical Analysis")
    numerical_analysis = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        numerical_analysis[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
    
    # 6. Correlation Analysis
    print("🔗 6. Correlation Analysis")
    correlation_matrix = df.corr(numeric_only=True) if len(numerical_cols) > 0 else pd.DataFrame()
    
    # 7. Generate Visualizations
    print("📊 7. Generating Visualizations")
    
    # Distribution plots for numerical columns
    for col in numerical_cols[:5]:  # Limit to first 5 numerical columns
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df[col].hist(bins=30, alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        df[col].plot(kind='box')
        plt.title(f'Box Plot of {col}')
        plt.ylabel(col)
        
        plt.tight_layout()
        plt.savefig(f'data/eda_results/health_dataset/plots/{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Categorical column analysis
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        plt.figure(figsize=(12, 6))
        value_counts = df[col].value_counts()
        
        plt.subplot(1, 2, 1)
        value_counts.plot(kind='bar')
        plt.title(f'Value Counts for {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        value_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.savefig(f'data/eda_results/health_dataset/plots/{col}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Correlation heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        plt.savefig('data/eda_results/health_dataset/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Pair plot for key numerical variables (limit to 4 variables to avoid overcrowding)
    if len(numerical_cols) >= 2:
        key_numerical_cols = numerical_cols[:4]  # Take first 4 numerical columns
        plt.figure(figsize=(12, 8))
        sns.pairplot(df[key_numerical_cols])
        plt.suptitle('Pair Plot of Key Numerical Variables', y=1.02)
        plt.tight_layout()
        plt.savefig('data/eda_results/health_dataset/plots/pair_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. Advanced Analysis
    print("🔬 8. Advanced Analysis")
    
    # Outlier detection
    outliers_analysis = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outliers_analysis[col] = {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    # 9. Summary Statistics
    print("📋 9. Summary Statistics")
    summary_stats = {
        'total_patients': len(df),
        'total_features': len(df.columns),
        'numerical_features': len(numerical_cols),
        'categorical_features': len(categorical_cols),
        'missing_values_total': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_quality_score': calculate_data_quality_score(df)
    }
    
    # 10. Compile all results
    eda_results = {
        'dataset_info': dataset_info,
        'basic_stats': basic_stats,
        'data_quality': data_quality,
        'categorical_analysis': categorical_analysis,
        'numerical_analysis': numerical_analysis,
        'correlation_matrix': correlation_matrix.to_dict(),
        'outliers_analysis': outliers_analysis,
        'summary_stats': summary_stats,
        'analysis_date': datetime.now().isoformat()
    }
    
    # Save results
    with open('data/eda_results/health_dataset/comprehensive_eda_results.json', 'w') as f:
        json.dump(eda_results, f, indent=4, default=str)
    
    print("✅ Comprehensive EDA completed and saved")
    return eda_results

def calculate_data_quality_score(df):
    """Calculate a data quality score (0-100)"""
    score = 100
    
    # Penalize for missing values
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    score -= missing_percentage * 2
    
    # Penalize for duplicates
    duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
    score -= duplicate_percentage * 2
    
    # Penalize for constant columns
    constant_cols = df.nunique() == 1
    constant_percentage = (constant_cols.sum() / len(df.columns)) * 100
    score -= constant_percentage * 5
    
    return max(0, min(100, score))

def generate_eda_report(eda_results):
    """Generate a comprehensive EDA report"""
    print("📄 Generating EDA Report...")
    
    report = f"""
# HealthAI Dataset - Comprehensive EDA Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Total Patients**: {eda_results['summary_stats']['total_patients']}
- **Total Features**: {eda_results['summary_stats']['total_features']}
- **Numerical Features**: {eda_results['summary_stats']['numerical_features']}
- **Categorical Features**: {eda_results['summary_stats']['categorical_features']}
- **Data Quality Score**: {eda_results['summary_stats']['data_quality_score']:.1f}/100

## Data Quality Assessment
- **Missing Values**: {eda_results['summary_stats']['missing_values_total']}
- **Duplicate Rows**: {eda_results['summary_stats']['duplicate_rows']}

## Key Findings
"""
    
    # Add key findings based on analysis
    if eda_results['summary_stats']['data_quality_score'] > 80:
        report += "- ✅ **Excellent data quality** - Ready for analysis\n"
    elif eda_results['summary_stats']['data_quality_score'] > 60:
        report += "- ⚠️ **Good data quality** - Minor cleaning needed\n"
    else:
        report += "- ❌ **Poor data quality** - Significant cleaning required\n"
    
    # Add insights about missing values
    high_missing_cols = [col for col, missing in eda_results['data_quality']['missing_percentage'].items() if missing > 20]
    if high_missing_cols:
        report += f"- ⚠️ **High missing values** in: {', '.join(high_missing_cols)}\n"
    
    # Add insights about outliers
    high_outlier_cols = [col for col, outlier_info in eda_results['outliers_analysis'].items() if outlier_info['outlier_percentage'] > 10]
    if high_outlier_cols:
        report += f"- ⚠️ **High outlier percentage** in: {', '.join(high_outlier_cols)}\n"
    
    report += """
## Recommendations
1. **Data Cleaning**: Address missing values and outliers
2. **Feature Engineering**: Create new features from existing ones
3. **Model Selection**: Choose appropriate algorithms based on data characteristics
4. **Validation**: Implement robust cross-validation strategies

## Generated Files
- `comprehensive_eda_results.json` - Complete analysis results
- `plots/` - Directory containing all visualizations
- `eda_report.md` - This report
"""
    
    # Save report
    with open('data/eda_results/health_dataset/eda_report.md', 'w') as f:
        f.write(report)
    
    print("✅ EDA report generated")
    return report

def main():
    """Run comprehensive EDA"""
    print("🚀 Starting Comprehensive EDA for HealthAI Dataset")
    print("=" * 60)
    
    # Load dataset
    print("\n📁 Loading Dataset...")
    df = load_health_dataset()
    if df is None:
        print("❌ Cannot proceed without dataset")
        return
    
    # Perform comprehensive EDA
    print("\n📊 Performing Comprehensive EDA...")
    eda_results = perform_comprehensive_eda(df)
    
    # Generate report
    print("\n📄 Generating EDA Report...")
    report = generate_eda_report(eda_results)
    
    # Summary
    print("\n📋 EDA Summary")
    print("=" * 60)
    print(f"✅ Dataset analyzed: {eda_results['summary_stats']['total_patients']} patients")
    print(f"✅ Features analyzed: {eda_results['summary_stats']['total_features']}")
    print(f"✅ Data quality score: {eda_results['summary_stats']['data_quality_score']:.1f}/100")
    print(f"✅ Visualizations generated: Check data/eda_results/health_dataset/plots/")
    print(f"✅ Report generated: data/eda_results/health_dataset/eda_report.md")
    
    print("\n🎉 Comprehensive EDA completed successfully!")
    print("📁 Check the following directories for results:")
    print("   - data/eda_results/health_dataset/ - Complete EDA results")
    print("   - data/eda_results/health_dataset/plots/ - All visualizations")
    print("   - data/eda_results/health_dataset/eda_report.md - EDA report")

if __name__ == "__main__":
    main()