import os
import sys
import streamlit as st
import requests
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components and pages
from ui.components.sidebar import show_sidebar
from ui.components.patient_context import render_patient_selector
from ui.data_display import show_data_display
from ui.eda_visualization import show_eda_visualization
from ui.model_results import show_model_results
from ui.ai_tools_demo import show_ai_tools_demo
from ui.disease_prediction import show_disease_prediction
from ui.los_prediction import show_los_prediction
from ui.patient_cohorts import show_patient_cohorts
from ui.ai_chat_bot import show_ai_chat_bot
from ui.tool_risk_assessment import show_risk_assessment
from ui.tool_notes_summarizer import show_notes_summarizer
from ui.tool_image_diagnostics import show_image_diagnostics
from ui.tool_feedback_analysis import show_feedback_analysis

st.set_page_config(page_title="HealthAI Dashboard", layout="wide")

# Get selected page and API base from sidebar
selected_page, API_BASE = show_sidebar() # sidebar now handles patient context internally

def parse_features(text: str):
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
        return vals
    except Exception:
        return None

# Main content area
if selected_page == "home":
    st.title("🏥 HealthAI Dashboard")
    
    # Check for current patient context
    current_patient = st.session_state.get('current_patient')
    
    if current_patient:
        # Patient Snapshot View
        st.success(f"👤 Currently Viewing: **{current_patient.get('name', 'Unknown')}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📋 Demographics")
            st.write(f"**Patient ID:** `{current_patient.get('patient_id')}`")
            st.write(f"**Age:** {current_patient.get('age')} years")
            st.write(f"**Gender:** {current_patient.get('gender')}")
            st.write(f"**Contact:** {current_patient.get('contact', 'N/A')}")
            
        with col2:
            st.markdown("### 🩺 Vitals Snapshot")
            c_bp, c_hr = st.columns(2)
            c_bp.metric("Blood Pressure", f"{current_patient.get('vitals_bp', '-')}")
            c_hr.metric("Heart Rate", f"{current_patient.get('vitals_hr', '-')} bpm")
            
            c_temp, c_spo2 = st.columns(2)
            c_temp.metric("Temp", f"{current_patient.get('temperature', '-')} °F")
            c_spo2.metric("SpO2", f"{current_patient.get('oxygen_saturation', '-')} %")
            
        with col3:
            st.markdown("### ⚠️ Clinical Status")
            # In a real app, this would fetch the latest prediction
            st.info("No recent alerts.")
            st.progress(0.2, text="Risk Score: Low (Estimated)")
            
        # st.markdown("---")
        # st.subheader("🚀 Quick Actions")
        # qa_col1, qa_col2, qa_col3 = st.columns(3)
        # with qa_col1:
        #     if st.button("🏥 Run Health Analysis", use_container_width=True):
        #         st.switch_page("ui/dashboard.py") # Ideally switch to specific page/tab
        #         st.write("Navigate to AI Tools > Health Analysis")
        # with qa_col2:
        #     if st.button("🔮 Predict Length of Stay", use_container_width=True):
        #         st.write("Navigate to LOS Prediction")
        
    else:
        # System Overview View (No Patient Selected)
        st.markdown("### 📊 System Overview")
        
        stats_c1, stats_c2, stats_c3, stats_c4 = st.columns(4)
        stats_c1.metric("Total Patients", "42", "+3 this week")
        stats_c2.metric("High Risk Cases", "5", "-1")
        stats_c3.metric("AI Predictions", "1,204", "+150")
        stats_c4.metric("System Status", "Online", "✅")
        
        st.info("👈 **Get Started:** Select a patient from the sidebar 'Patient Context' to view their profile and run analyses.")
        
        st.markdown("### 🗓️ Recent Activity")
        st.dataframe({
            "Time": ["10:30 AM", "09:15 AM", "Yesterday"],
            "Event": ["New Patient Admitted", "Alert: High HR", "System Backup"],
            "User": ["Dr. Smith", "System", "Admin"]
        }, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Admit Patient (Collapsible)
    with st.expander("➕ Admit New Patient"):
        with st.form("admit_patient_form_home"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full Name")
                gender = st.selectbox("Gender", ["M", "F", "Other"])
                age = st.number_input("Age", 0, 120, 30)
            with c2:
                dob = st.date_input("Date of Birth")
                address = st.text_input("Address")
                contact = st.text_input("Contact Number")
                
            submit = st.form_submit_button("Admit Patient")
            if submit:
                payload = {
                    "name": name, "gender": gender, "age": int(age),
                    "dob": str(dob), "address": address, "contact": contact
                }
                try:
                    res = requests.post(f"{API_BASE}/patients", json=payload, timeout=10)
                    if res.status_code == 200:
                        st.success(f"✅ Patient Admitted! ID: {res.json()['patient_id']}")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to admit patient: {e}")

    # Patient Lookup (Collapsible)
    with st.expander("🔍 Lookup Patient by ID"):
        l_col1, l_col2 = st.columns([3, 1])
        with l_col1:
            lookup_id = st.text_input("Patient ID", key="lookup_patient_id_home", label_visibility="collapsed", placeholder="Enter Patient ID here...")
        with l_col2:
            if st.button("Load Profile", key="btn_lookup_home"):
                try:
                    res = requests.get(f"{API_BASE}/patients/{lookup_id}", timeout=10)
                    if res.status_code == 200:
                        st.session_state['current_patient'] = res.json()
                        st.success("Loaded!")
                        st.experimental_rerun()
                    else:
                        st.error("Not found.")
                except Exception as e:
                    st.error(f"Error: {e}")

elif selected_page == "disease_prediction":
    show_disease_prediction(API_BASE)

elif selected_page == "los_prediction":
    show_los_prediction(API_BASE)

elif selected_page == "patient_cohorts":
    show_patient_cohorts(API_BASE)

elif selected_page == "data_display":
    show_data_display()

elif selected_page == "eda_visualization":
    show_eda_visualization()

elif selected_page == "model_results":
    show_model_results()

elif selected_page == "ai_tools_demo":
    show_ai_tools_demo()

elif selected_page == "ai_tool_ai_chat_bot":
    show_ai_chat_bot()

elif selected_page == "ai_tool_risk_assessment":
    show_risk_assessment(API_BASE)

elif selected_page == "ai_tool_notes_summarizer":
    show_notes_summarizer(API_BASE)

elif selected_page == "ai_tool_image_diagnostics":
    show_image_diagnostics(API_BASE)

elif selected_page == "ai_tool_feedback_analysis":
    show_feedback_analysis(API_BASE)

elif selected_page == "ai_tools_demo":
    # Keep generic demo page as fallback or specific link
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
    - **HealthAI Assistant**: Interactive AI chatbot
    - **AI Tools Demo**: Advanced AI utilities
    
    ### API Configuration
    The API base URL can be configured in the sidebar. Default is `http://localhost:8000`.
    """)

else:
    # Default or fallback
    st.title("🏥 HealthAI Dashboard")
    st.info("Select a page from the sidebar to get started.")