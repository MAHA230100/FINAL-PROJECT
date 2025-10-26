import os
import sys
import streamlit as st
import requests
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components and pages
from ui.components.sidebar import show_sidebar
from ui.pages.data_display import show_data_display
from ui.pages.eda_visualization import show_eda_visualization
from ui.pages.model_results import show_model_results
from ui.pages.ai_tools_demo import show_ai_tools_demo

st.set_page_config(page_title="HealthAI Dashboard", layout="wide")

# Get selected page and API base from sidebar
selected_page, API_BASE = show_sidebar()

def parse_features(text: str):
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
        return vals
    except Exception:
        return None

# Main content area
if selected_page == "home":
    st.title("🏥 HealthAI Dashboard")
    st.markdown("Welcome to the HealthAI platform for healthcare data science and AI tools.")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Models", "3", "1")
    with col2:
        st.metric("Datasets", "5", "2")
    with col3:
        st.metric("Predictions Today", "127", "23")
    with col4:
        st.metric("System Status", "Online", "✅")
    
    st.markdown("---")
    
    # Quick access tabs
    st.subheader("Quick Access")
    tabs = st.tabs([
        "🔬 Disease Prediction",
        "📈 LOS Prediction", 
        "👥 Patient Clustering",
        "🛠️ AI Tools"
    ])
    
    with tabs[0]:
        st.subheader("Disease Risk Classification")
        features = st.text_input("Enter features (comma-separated)", key="cls_features")
        if st.button("Predict risk", key="predict_risk"):
            vals = parse_features(features)
            if not vals:
                st.error("Provide numeric features, comma-separated")
            else:
                try:
                    res = requests.post(f"{API_BASE}/predict/classify", json={"features": vals}, timeout=10)
                    st.json(res.json())
                except Exception as e:
                    st.error(f"Request failed: {e}")

    with tabs[1]:
        st.subheader("Length of Stay Prediction")
        features = st.text_input("Enter features (comma-separated)", key="reg_features")
        if st.button("Predict LOS", key="predict_los"):
            vals = parse_features(features)
            if not vals:
                st.error("Provide numeric features, comma-separated")
            else:
                try:
                    res = requests.post(f"{API_BASE}/predict/regress", json={"features": vals}, timeout=10)
                    st.json(res.json())
                except Exception as e:
                    st.error(f"Request failed: {e}")

    with tabs[2]:
        st.subheader("Patient Clustering")
        st.info("Stub: visualize clusters and profiles")

    with tabs[3]:
        st.subheader("AI Tools")
        st.info("Quick access to AI tools and utilities")

elif selected_page == "disease_prediction":
    st.title("🔬 Disease Prediction")
    st.subheader("Disease Risk Classification")
    features = st.text_input("Enter features (comma-separated)", key="cls_features_main")
    if st.button("Predict risk", key="predict_risk_main"):
        vals = parse_features(features)
        if not vals:
            st.error("Provide numeric features, comma-separated")
        else:
            try:
                res = requests.post(f"{API_BASE}/predict/classify", json={"features": vals}, timeout=10)
                st.json(res.json())
            except Exception as e:
                st.error(f"Request failed: {e}")

elif selected_page == "los_prediction":
    st.title("📈 LOS Prediction")
    st.subheader("Length of Stay Prediction")
    features = st.text_input("Enter features (comma-separated)", key="reg_features_main")
    if st.button("Predict LOS", key="predict_los_main"):
        vals = parse_features(features)
        if not vals:
            st.error("Provide numeric features, comma-separated")
        else:
            try:
                res = requests.post(f"{API_BASE}/predict/regress", json={"features": vals}, timeout=10)
                st.json(res.json())
            except Exception as e:
                st.error(f"Request failed: {e}")

elif selected_page == "patient_cohorts":
    st.title("👥 Patient Cohorts")
    st.subheader("Patient Clustering")
    st.info("Stub: visualize clusters and profiles")

elif selected_page == "data_display":
    show_data_display()

elif selected_page == "eda_visualization":
    show_eda_visualization()

elif selected_page == "model_results":
    show_model_results()

elif selected_page == "ai_tools_demo":
    show_ai_tools_demo()

elif selected_page == "help":
    st.title("❓ Help")
    st.markdown("""
    ## HealthAI Platform Help
    
    ### Navigation
    - **Home**: Main dashboard with quick access to key features
    - **Disease Prediction**: Risk classification for diseases
    - **LOS Prediction**: Length of stay prediction
    - **Patient Cohorts**: Patient clustering and segmentation
    - **Data Display**: View raw and cleaned datasets
    - **EDA Visualization**: Exploratory data analysis
    - **Model Results**: View model performance and metrics
    - **AI Tools Demo**: AI utilities and tools
    
    ### API Configuration
    The API base URL can be configured in the sidebar. Default is `http://localhost:8000`.
    
    ### Getting Started
    1. Ensure the API server is running
    2. Configure the API base URL if needed
    3. Navigate to the desired page using the sidebar
    4. Follow the on-page instructions for each feature
    """)

else:
    st.title("🏥 HealthAI Dashboard")
    st.info("Select a page from the sidebar to get started.") 