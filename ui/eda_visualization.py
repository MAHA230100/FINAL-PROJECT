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
    eda_results_path = Path("data/eda_results")
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

    # Visualization tabs
    st.subheader("📈 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Correlations", "Outcomes", "Interactive"])
    
    with tab1:
        st.write("### Data Distributions")
        
        # Check for distribution plots in the health_dataset/plots directory
        plots_path = eda_results_path / "health_dataset" / "plots"
        if plots_path.exists():
            plot_files = sorted(os.listdir(plots_path))
            found_plots = False
            
            # Show distribution plots for numerical variables
            for plot_file in plot_files:
                if plot_file.endswith('_distribution.png'):
                    var_name = plot_file.replace('_distribution.png', '').replace('_', ' ').title()
                    st.image(str(plots_path / plot_file), 
                            caption=f"{var_name} Distribution and Box Plot")
                    found_plots = True
            
            # Show categorical analysis plots
            for plot_file in plot_files:
                if plot_file.endswith('_analysis.png'):
                    var_name = plot_file.replace('_analysis.png', '').replace('_', ' ').title()
                    st.image(str(plots_path / plot_file), 
                            caption=f"{var_name} Analysis")
                    found_plots = True
            
            if not found_plots:
                st.info("No distribution plots found. Run EDA analysis to generate them.")
        else:
            st.info("Plots directory not found. Run EDA analysis to generate them.")
    
    with tab2:
        st.write("### Correlation Analysis")
        
        plots_path = eda_results_path / "health_dataset" / "plots"
        if plots_path.exists():
            plot_files = os.listdir(plots_path)
            found_heatmap = False
            found_pairplot = False
            
            # Show correlation heatmap if it exists
            for plot_file in plot_files:
                if 'correlation_heatmap' in plot_file.lower():
                    st.image(str(plots_path / plot_file), 
                            caption="Correlation Heatmap of Numerical Variables")
                    found_heatmap = True
                    break
            
            # Show pairplot if it exists
            for plot_file in plot_files:
                if 'pairplot' in plot_file.lower():
                    st.image(str(plots_path / plot_file), 
                            caption="Pairwise Relationships")
                    found_pairplot = True
                    break
            
            if not (found_heatmap or found_pairplot):
                st.info("No correlation plots found. Run EDA analysis to generate them.")
        else:
            st.info("Plots directory not found. Run EDA analysis to generate them.")
    
    with tab3:
        st.write("### Data Analysis")
        
        plots_path = eda_results_path / "health_dataset" / "plots"
        if plots_path.exists():
            plot_files = os.listdir(plots_path)
            found_plots = False
            
            # Show any remaining plots that haven't been shown yet
            for plot_file in plot_files:
                if (not plot_file.endswith('_distribution.png') and 
                    not plot_file.endswith('_analysis.png') and
                    'correlation' not in plot_file.lower() and
                    'pairplot' not in plot_file.lower() and
                    plot_file.endswith('.png')):
                    
                    caption = plot_file.replace('_', ' ').replace('.png', '').title()
                    st.image(str(plots_path / plot_file), caption=caption)
                    found_plots = True
            
            if not found_plots:
                st.info("No additional analysis plots found. Check other tabs for visualizations.")
        else:
            st.info("Plots directory not found. Run EDA analysis to generate them.")
    
    # Interactive Visualizations Tab
    with tab4:
        st.write("### Interactive Visualizations")
        
        plots_path = eda_results_path / "health_dataset" / "plots"
        if plots_path.exists():
            plot_files = os.listdir(plots_path)
            interactive_files = [f for f in plot_files if f.endswith('.html')]
            
            if interactive_files:
                st.write("**Available Interactive Plots:**")
                
                # Display a dropdown to select which plot to view
                selected_plot = st.selectbox(
                    "Select a plot to view:",
                    interactive_files,
                    format_func=lambda x: x.replace('_', ' ').replace('.html', '').title()
                )
                
                # Display the selected interactive plot
                with open(plots_path / selected_plot, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=600, scrolling=True)
            else:
                st.info("No interactive plots found. Interactive visualizations will be available in a future update.")
                
                # Optional: Add a button to generate sample interactive plots
                if st.button("Generate Sample Interactive Plot"):
                    try:
                        # Create a sample interactive plot using Plotly
                        import plotly.express as px
                        from plotly.subplots import make_subplots
                        
                        # Sample data
                        np.random.seed(42)
                        df = pd.DataFrame({
                            'Age': np.random.normal(45, 15, 1000),
                            'BloodPressure': np.random.normal(120, 20, 1000),
                            'HeartRate': np.random.normal(75, 10, 1000),
                            'RiskScore': np.random.uniform(0, 1, 1000)
                        })
                        
                        # Create an interactive scatter plot
                        fig = px.scatter(
                            df, x='Age', y='BloodPressure', 
                            color='RiskScore',
                            title='Sample Interactive Plot: Age vs Blood Pressure',
                            labels={'Age': 'Age (years)', 'BloodPressure': 'Blood Pressure (mmHg)'},
                            color_continuous_scale='Viridis'
                        )
                        
                        # Save as HTML
                        os.makedirs(plots_path, exist_ok=True)
                        fig.write_html(str(plots_path / 'sample_interactive_plot.html'))
                        st.success("Sample interactive plot generated! Refresh the page to view it.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to generate sample plot: {str(e)}")
        else:
            st.info("Plots directory not found. Run EDA analysis to generate visualizations.")

    # Action buttons
    # st.subheader("🔧 Actions")
    
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     if st.button("🔄 Refresh EDA Results"):
    #         st.rerun()
    
    # with col2:
    #     if st.button("📊 Run New EDA Analysis"):
    #         st.info("Run: `python scripts/run_comprehensive_eda.py` in terminal")
    
    # with col3:
    #     if st.button("💾 Download Results"):
    #         st.info("EDA results are saved in the 'eda_results' directory")


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