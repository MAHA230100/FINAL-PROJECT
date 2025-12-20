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
        # Standardized Header
        st.info(f"👤 Patient Context: **{current_patient.get('name', 'Unknown')}** (ID: {current_patient.get('patient_id')})")
        
        # Main Snapshot Card
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 1.5, 1])
            
            with col1:
                st.markdown("### 📋 Demographics")
                st.write(f"**Age:** {current_patient.get('age', 'N/A')}y")
                st.write(f"**Gender:** {current_patient.get('gender', 'N/A')}")
                st.write(f"**Condition:** {current_patient.get('medical_condition', 'N/A')}")
                st.write(f"**Infection Risk:** {current_patient.get('infection_risk', 'N/A')}")
                
            with col2:
                st.markdown("### 🩺 Latest Vitals")
                v_col1, v_col2 = st.columns(2)
                v_col1.metric("Blood Pressure", f"{current_patient.get('vitals_bp', '-')}", help="Latest Systolic BP")
                v_col2.metric("Heart Rate", f"{current_patient.get('vitals_hr', '-')} bpm")
                
                v_col3, v_col4 = st.columns(2)
                v_col3.metric("Temp", f"{current_patient.get('temperature', '-')} °F")
                v_col4.metric("SpO2", f"{current_patient.get('oxygen_saturation', '-')} %")
                
            with col3:
                st.markdown("### ⚠️ AI Insights")
                # Dynamic risk calculation for display
                try:
                    risk_val = float(current_patient.get('risk_score', 0.2))
                except:
                    risk_val = 0.2
                    
                risk_status = "High" if risk_val > 0.7 else "Medium" if risk_val > 0.4 else "Low"
                risk_color = "red" if risk_status == "High" else "orange" if risk_status == "Medium" else "green"
                
                st.markdown(f"**Risk Level:** :{risk_color}[{risk_status}]")
                st.progress(min(1.0, max(0.0, risk_val)), text=f"System Risk Score: {risk_val:.2f}")
                
                if st.button("Generate Detailed Prognosis", use_container_width=True):
                    st.toast("Analyzing latest clinical data...")
                    st.info("Prognosis: Patient stable, monitor vitals every 4 hours.")
            
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
        st.markdown("### Patient Information")
        with st.form("admit_patient_form_home", border=False):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full Name", placeholder="e.g. John Doe")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                age = st.number_input("Age", 0, 120, 30)
            with c2:
                dob = st.date_input("Date of Birth")
                address = st.text_input("Address", placeholder="Full Home Address")
                contact = st.text_input("Contact Number", placeholder="+1 (555) 000-0000")
            
            st.markdown("---")
            am_c1, am_c2 = st.columns([2, 1])
            with am_c2:
                submit = st.form_submit_button("Confirm Admission", type="primary", use_container_width=True)
            
            if submit:
                if not name or not address or not contact:
                    st.error("Please fill in all required fields (Name, Address, Contact).")
                else:
                    payload = {
                        "name": name, 
                        "gender": "M" if gender == "Male" else "F" if gender == "Female" else "Other", 
                        "age": int(age),
                        "dob": str(dob), 
                        "address": address, 
                        "contact": contact,
                        "medical_condition": "New Admission",
                        "admission_type": "Other"
                    }
                    try:
                        res = requests.post(f"{API_BASE}/patients", json=payload, timeout=10)
                        if res.status_code == 200:
                            p_id = res.json().get('patient_id')
                            st.success(f"🎉 Patient Admitted Successfully! Assigned ID: `{p_id}`")
                            st.balloons()
                        else:
                            st.error(f"Error from API: {res.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to API: {e}")

    # Patient Lookup (Collapsible)
    with st.expander("🔍 Lookup Patient by ID"):
        # detection of ID change to reset preview
        if "last_lookup_id" not in st.session_state:
            st.session_state.last_lookup_id = ""
        
        l_col1, l_col2, l_col3 = st.columns([3, 1, 1])
        with l_col1:
            lookup_id = st.text_input("Patient ID", key="lookup_patient_id_home", label_visibility="collapsed", placeholder="Enter Patient ID (e.g., HCXXXXXX)")
            if lookup_id != st.session_state.last_lookup_id:
                st.session_state['temp_lookup_patient'] = None
                st.session_state.last_lookup_id = lookup_id
                
        with l_col2:
            search_clicked = st.button("Search Profile", key="btn_lookup_search", use_container_width=True)
        with l_col3:
            if st.button("Clear", key="btn_lookup_clear", use_container_width=True):
                st.session_state['temp_lookup_patient'] = None
                st.session_state.last_lookup_id = ""
                st.rerun()

        if search_clicked:
            if not lookup_id:
                st.warning("Please enter a Patient ID.")
            else:
                try:
                    res = requests.get(f"{API_BASE}/patients/{lookup_id}", timeout=10)
                    if res.status_code == 200:
                        st.session_state['temp_lookup_patient'] = res.json()
                    else:
                        st.session_state['temp_lookup_patient'] = None
                        st.error(f"Patient ID `{lookup_id}` not found.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

        # Show Preview and Activate Button
        temp_patient = st.session_state.get('temp_lookup_patient')
        if temp_patient:
            st.markdown("---")
            st.success("✅ Patient Profile Found")
            
            # Modern Profile Card
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    # Dynamic avatar emoji
                    avatar = "👨" if temp_patient.get('gender', 'M').upper() == 'M' else "👩"
                    st.markdown(f"<div style='text-align: center; font-size: 80px;'>{avatar}</div>", unsafe_allow_html=True)
                
                with c2:
                    st.subheader(temp_patient.get('name', 'Unknown'))
                    st.write(f"**Patient ID:** `{temp_patient.get('patient_id')}`")
                    
                    # Quick Metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Age", f"{temp_patient.get('age')}y")
                    m_col2.metric("Gender", temp_patient.get('gender'))
                    m_col3.metric("Insurance", temp_patient.get('insurance', 'N/A'))
            
            # Clinical Snapshot Details
            with st.container(border=True):
                st.markdown("**Clinical Details**")
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.write(f"**Medical Condition:** {temp_patient.get('medical_condition', 'N/A')}")
                    st.write(f"**Admission Type:** {temp_patient.get('admission_type', 'N/A')}")
                with d_col2:
                    st.write(f"**Contact:** {temp_patient.get('contact', 'N/A')}")
                    st.write(f"**Admitted:** {temp_patient.get('date_of_admission', 'N/A')}")
            
            if st.button("🚀 Set as Active Patient Context", key="btn_activate_lookup", type="primary", use_container_width=True):
                st.session_state['current_patient'] = temp_patient
                st.session_state['temp_lookup_patient'] = None
                st.rerun()

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