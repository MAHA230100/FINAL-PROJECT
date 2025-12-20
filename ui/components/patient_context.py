import streamlit as st
import requests
from typing import Optional, Dict, Any

def get_all_patients(api_base: str) -> list:
    """Fetch all patients for the dropdown."""
    try:
        resp = requests.get(f"{api_base}/patients")
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []

def render_patient_selector(api_base: str):
    """
    Renders a selectbox in the sidebar to choose a patient.
    Updates st.session_state['current_patient'] with the full patient dict.
    """
    st.sidebar.markdown("### 👤 Patient Context")
    
    # 1. Load patient list
    patients = get_all_patients(api_base)
    if not patients:
        st.sidebar.warning("No patients found in DB.")
        # Don't reset current_patient, just return. 
        # This prevents flaking API from wiping context.
        return

    # 2. Prepare options
    # Format: "Name (ID)"
    patient_options = {f"{p['name']} ({p['patient_id']})": p for p in patients}
    
    # 3. Determine default index
    current = st.session_state.get('current_patient')
    index = 0
    if current:
        current_label = f"{current['name']} ({current['patient_id']})"
        if current_label in patient_options:
            keys = list(patient_options.keys())
            index = keys.index(current_label)

    # 4. Selectbox
    # Note: We rely on the 'index' parameter and external callbacks (like in dashboard.py) 
    # to sync this widget. Manually overwriting the session state key here can 
    # interfere with manual user selection in the sidebar.
    selectbox_key = "patient_selector_box"
    
    selected_label = st.sidebar.selectbox(
        "Select Patient",
        options=list(patient_options.keys()),
        index=index,
        key=selectbox_key
    )
    
    # 5. Update session state
    if selected_label:
        new_patient = patient_options[selected_label]
        # Only update if changed to avoid unnecessary re-runs or state churn
        if not current or current.get('patient_id') != new_patient.get('patient_id'):
            st.session_state['current_patient'] = new_patient
        
        # Display mini info
        st.sidebar.info(f"**Age:** {new_patient.get('age')} | **Gender:** {new_patient.get('gender')}")
