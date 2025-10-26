import streamlit as st
import os

def show_sidebar():
    """Create the sidebar navigation with improved layout"""
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .status-online {
            color: #00cc00;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
        }
        .status-dot {
            height: 10px;
            width: 10px;
            background-color: #00cc00;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .nav-item {
            padding: 8px 0;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        .nav-item:hover {
            background-color: #f0f2f6;
        }
        .ai-tool {
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .ai-tool:hover {
            background-color: #f0f2f6;
            transform: translateX(4px);
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar header with status
    st.sidebar.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
        <h1 style='margin: 0;'>🏥 HealthAI</h1>
        <div class="status-online">
            <span class="status-dot"></span>
            <span>Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Configuration - moved below navigation
    _default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    
    # Main pages with icons and better spacing
    pages = {
        "🏠 Home": "home",
        "📊 Data Display": "data_display",
        "📈 EDA Visualization": "eda_visualization", 
        "🤖 Model Results": "model_results",
        "🔬 Disease Prediction": "disease_prediction", 
        "📈 LOS Prediction": "los_prediction",
        "👥 Patient Cohorts": "patient_cohorts",
        "❓ Help": "help"
    }
    
    # Create navigation items with better styling
    selected_page = None
    for page_name, page_id in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_id}", use_container_width=True):
            selected_page = page_id
    
    # If no page selected (first load), default to home
    if selected_page is None:
        selected_page = pages["🏠 Home"]
    
    st.sidebar.markdown("---")
    
    # AI Tools section with clickable items
    st.sidebar.markdown("### AI Tools")
    ai_tools = [
        ("🔍", "Patient Risk Assessment", "risk_assessment"),
        ("📝", "Notes Summarizer", "notes_summarizer"), 
        ("🖼️", "Image Diagnostics", "image_diagnostics"),
        ("💬", "Feedback Analysis", "feedback_analysis"),
        ("⚙️", "Admin Dashboard", "admin_dashboard")
    ]
    
    for icon, name, key in ai_tools:
        if st.sidebar.button(
            f"{icon} {name}", 
            key=f"ai_tool_{key}", 
            use_container_width=True,
            help=f"Open {name} tool"
        ):
            selected_page = f"ai_tool_{key}"
    
    st.sidebar.markdown("---")
    
    # API Configuration at the bottom
    api_base = st.sidebar.text_input(
        "API Base URL", 
        _default_api, 
        key="api_base_url_global",
        help="Enter the base URL for the API endpoints"
    )
    
    return selected_page, api_base
