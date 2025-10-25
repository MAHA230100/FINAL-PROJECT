import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show_eda_visualization():
    """EDA visualization page"""
    st.header("📈 EDA Visualization")
    
    _default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_BASE = st.sidebar.text_input("API base URL", _default_api, key="api_base_url_eda")
    
    # EDA Analysis Section
    st.subheader("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.text_input("Dataset Name", value="mimic_demo", key="eda_dataset")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["basic", "advanced", "statistical"],
            key="eda_analysis_type"
        )
    
    with col2:
        if st.button("Run EDA Analysis", key="run_eda"):
            try:
                response = requests.post(
                    f"{API_BASE}/eda/analyze",
                    json={"dataset_name": dataset_name, "analysis_type": analysis_type},
                    timeout=30
                )
                result = response.json()
                st.success("EDA analysis completed!")
                st.json(result)
            except Exception as e:
                st.error(f"EDA analysis failed: {e}")
    
    # Visualizations Section
    st.subheader("Available Visualizations")
    
    if st.button("Load Visualizations", key="load_viz"):
        try:
            response = requests.get(f"{API_BASE}/eda/visualizations", timeout=10)
            viz_data = response.json()
            
            if "visualizations" in viz_data:
                st.write("Available visualizations:")
                for viz in viz_data["visualizations"]:
                    st.write(f"- **{viz['name']}**: {viz['type']}")
        except Exception as e:
            st.error(f"Failed to load visualizations: {e}")
    
    # Sample visualizations (placeholder)
    st.subheader("Sample Visualizations")
    
    # Create sample data for demonstration
    import numpy as np
    
    # Sample distribution plot
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
