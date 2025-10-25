import streamlit as st
import requests
import pandas as pd
import os

def show_data_display():
    """Display raw and cleaned data"""
    st.header("📊 Data Display")
    
    _default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_BASE = st.sidebar.text_input("API base URL", _default_api, key="api_base_url_data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Data")
        if st.button("Load Raw Data", key="load_raw"):
            try:
                response = requests.get(f"{API_BASE}/data/raw", timeout=10)
                data = response.json()
                st.json(data)
                
                # Display as DataFrame if possible
                if "datasets" in data:
                    df = pd.DataFrame(data["datasets"])
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to load raw data: {e}")
    
    with col2:
        st.subheader("Cleaned Data")
        if st.button("Load Cleaned Data", key="load_cleaned"):
            try:
                response = requests.get(f"{API_BASE}/data/cleaned", timeout=10)
                data = response.json()
                st.json(data)
                
                # Display as DataFrame if possible
                if "datasets" in data:
                    df = pd.DataFrame(data["datasets"])
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to load cleaned data: {e}")
    
    # Data cleaning section
    st.subheader("Data Cleaning")
    dataset_name = st.text_input("Dataset Name", value="mimic_demo", key="clean_dataset")
    cleaning_options = {
        "remove_duplicates": st.checkbox("Remove Duplicates", value=True),
        "handle_missing": st.checkbox("Handle Missing Values", value=True),
        "normalize": st.checkbox("Normalize Data", value=False)
    }
    
    if st.button("Clean Data", key="clean_data"):
        try:
            response = requests.post(
                f"{API_BASE}/data/clean",
                json={"dataset_name": dataset_name, "cleaning_options": cleaning_options},
                timeout=30
            )
            result = response.json()
            st.success("Data cleaning completed!")
            st.json(result)
        except Exception as e:
            st.error(f"Data cleaning failed: {e}")
