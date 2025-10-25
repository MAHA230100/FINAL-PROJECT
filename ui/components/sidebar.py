import streamlit as st
import os

def show_sidebar():
    """Create the sidebar navigation"""
    st.sidebar.title("🏥 HealthAI")
    st.sidebar.markdown("---")
    
    # API Configuration
    _default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
    api_base = st.sidebar.text_input("API Base URL", _default_api, key="api_base_url_global")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader("📊 Navigation")
    
    # Main pages
    pages = {
        "🏠 Home": "home",
        "🔬 Disease Prediction": "disease_prediction", 
        "📈 LOS Prediction": "los_prediction",
        "👥 Patient Cohorts": "patient_cohorts",
        "📊 Data Display": "data_display",
        "📈 EDA Visualization": "eda_visualization", 
        "🤖 Model Results": "model_results",
        "🛠️ AI Tools Demo": "ai_tools_demo",
        "❓ Help": "help"
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Page",
        list(pages.keys()),
        key="page_selector"
    )
    
    st.sidebar.markdown("---")
    
    # AI Tools section
    st.sidebar.subheader("🛠️ AI Tools")
    ai_tools = [
        "🔍 Patient Risk Assessment",
        "📝 Notes Summarizer", 
        "🖼️ Image Diagnostics",
        "💬 Feedback Analysis",
        "⚙️ Admin Dashboard"
    ]
    
    for tool in ai_tools:
        st.sidebar.write(f"• {tool}")
    
    st.sidebar.markdown("---")
    
    # Status
    st.sidebar.subheader("📊 Status")
    st.sidebar.success("✅ System Online")
    
    return pages[selected_page], api_base
