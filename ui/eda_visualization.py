import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import os
import requests

def show_eda_visualization():
    st.title("📈 EDA Visualization")
    st.markdown("Comprehensive exploratory data analysis and visualization of healthcare insights.")
    
    # Check if EDA results exist
    eda_results_path = Path("eda_results")
    if not eda_results_path.exists():
        st.warning("⚠️ EDA results not found. Please run the EDA analysis first.")
        st.info("Run: `python scripts/run_comprehensive_eda.py` to generate EDA results.")
        
        # Show API-based EDA as fallback
        show_api_eda()
        return
    
    # Load summary statistics
    summary_path = eda_results_path / "summary_statistics.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Display dataset overview
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", summary['dataset_overview']['total_patients'])
        with col2:
            st.metric("Features", summary['dataset_overview']['total_features'])
        with col3:
            st.metric("Missing Values", summary['dataset_overview']['missing_values'])
        with col4:
            st.metric("Duplicate Rows", summary['dataset_overview']['duplicate_rows'])
    
    # Load clinical insights
    insights_path = eda_results_path / "clinical_insights.json"
    if insights_path.exists():
        with open(insights_path, 'r') as f:
            insights = json.load(f)
        
        st.subheader("🏥 Clinical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Patient Demographics**")
            st.write(f"• Average Age: {insights['patient_demographics']['average_age']:.1f} years")
            st.write(f"• Most Common Admission: {insights['patient_demographics']['most_common_admission_type']}")
            
            gender_dist = insights['patient_demographics']['gender_distribution']
            for gender, count in gender_dist.items():
                st.write(f"• {gender}: {count} patients")
        
        with col2:
            st.write("**Clinical Metrics**")
            st.write(f"• Average Length of Stay: {insights['clinical_metrics']['average_length_of_stay']:.1f} days")
            st.write(f"• Average Blood Pressure: {insights['clinical_metrics']['average_bp']:.1f}")
            st.write(f"• Average Heart Rate: {insights['clinical_metrics']['average_hr']:.1f}")
            st.write(f"• Average Lab Results: {insights['clinical_metrics']['average_lab_results']:.1f}")
    
    # Visualization tabs
    st.subheader("📈 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Correlations", "Outcomes", "Interactive"])
    
    with tab1:
        st.write("### Patient Demographics and Health Metrics")
        
        # Check for distribution plots
        dist_path = eda_results_path / "distributions"
        if dist_path.exists():
            if (dist_path / "demographics.png").exists():
                st.image(str(dist_path / "demographics.png"), caption="Patient Demographics")
            
            if (dist_path / "age_analysis.png").exists():
                st.image(str(dist_path / "age_analysis.png"), caption="Age Group Analysis")
        else:
            st.info("Distribution plots not found. Run EDA analysis to generate them.")
    
    with tab2:
        st.write("### Correlation Analysis")
        
        corr_path = eda_results_path / "correlations"
        if corr_path.exists():
            if (corr_path / "correlation_heatmap.png").exists():
                st.image(str(corr_path / "correlation_heatmap.png"), caption="Correlation Matrix")
            
            if (corr_path / "pairplot.png").exists():
                st.image(str(corr_path / "pairplot.png"), caption="Pairwise Relationships")
        else:
            st.info("Correlation plots not found. Run EDA analysis to generate them.")
    
    with tab3:
        st.write("### Outcome Analysis")
        
        outcomes_path = eda_results_path / "outcomes"
        if outcomes_path.exists():
            if (outcomes_path / "outcome_analysis.png").exists():
                st.image(str(outcomes_path / "outcome_analysis.png"), caption="Outcome Analysis")
            
            if (outcomes_path / "risk_analysis.png").exists():
                st.image(str(outcomes_path / "risk_analysis.png"), caption="Risk Analysis")
        else:
            st.info("Outcome analysis plots not found. Run EDA analysis to generate them.")
    
    with tab4:
        st.write("### Interactive Visualizations")
        
        interactive_path = eda_results_path / "interactive"
        if interactive_path.exists():
            interactive_files = list(interactive_path.glob("*.html"))
            
            if interactive_files:
                st.write("**Available Interactive Plots:**")
                for i, file in enumerate(interactive_files):
                    st.write(f"{i+1}. {file.stem.replace('_', ' ').title()}")
                
                # Display first interactive plot
                if (interactive_path / "age_vs_los_interactive.html").exists():
                    with open(interactive_path / "age_vs_los_interactive.html", 'r') as f:
                        st.components.v1.html(f.read(), height=600)
            else:
                st.info("Interactive plots not found. Run EDA analysis to generate them.")
        else:
            st.info("Interactive plots not found. Run EDA analysis to generate them.")
    
    # Risk Analysis Section
    if insights_path.exists():
        st.subheader("⚠️ Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Risk Patients", insights['outcome_analysis']['high_risk_patients'])
        
        with col2:
            readmission_rate = insights['outcome_analysis']['readmission_rate']
            st.metric("Readmission Rate", f"{readmission_rate:.1f}%")
        
        with col3:
            # Calculate risk correlation
            risk_correlations = insights['risk_factors']
            st.write("**Risk Correlations:**")
            st.write(f"• Age: {risk_correlations['age_risk_correlation']:.3f}")
            st.write(f"• Length of Stay: {risk_correlations['los_risk_correlation']:.3f}")
            st.write(f"• Blood Pressure: {risk_correlations['bp_risk_correlation']:.3f}")
    
    # Action buttons
    st.subheader("🔧 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh EDA Results"):
            st.rerun()
    
    with col2:
        if st.button("📊 Run New EDA Analysis"):
            st.info("Run: `python scripts/run_comprehensive_eda.py` in terminal")
    
    with col3:
        if st.button("💾 Download Results"):
            st.info("EDA results are saved in the 'eda_results' directory")


def show_api_eda():
    """Fallback to API-based EDA when local results are not available"""
    st.subheader("🌐 API-Based EDA Analysis")
    
    _default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_BASE = st.sidebar.text_input("API base URL", _default_api, key="api_base_url_eda")
    
    # EDA Analysis Section
    st.subheader("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.text_input("Dataset Name", value="healthcare_ai_dataset", key="eda_dataset")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["basic", "advanced", "statistical"],
            key="eda_analysis_type"
        )
    
    with col2:
        if st.button("Run EDA Analysis", key="run_eda"):
            try:
                response = requests.post(
                    f"{API_BASE}/eda/summary",
                    json={"dataset_name": dataset_name, "columns": [], "visualization_type": analysis_type},
                    timeout=30
                )
                result = response.json()
                st.success("EDA analysis completed!")
                st.json(result)
            except Exception as e:
                st.error(f"EDA analysis failed: {e}")
    
    # Sample visualizations (placeholder)
    st.subheader("Sample Visualizations")
    
    # Create sample data for demonstration
    st.write("**Distribution Plot**")
    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.random.normal(50, 15, 1000)
    ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Sample Data Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Sample correlation heatmap
    st.write("**Correlation Matrix**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_data = np.random.randn(100, 5)
    corr_matrix = np.corrcoef(sample_data.T)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Sample Correlation Matrix')
    st.pyplot(fig)