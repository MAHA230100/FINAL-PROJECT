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
        st.session_state['current_patient'] = None
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
    selected_label = st.sidebar.selectbox(
        "Select Patient",
        options=list(patient_options.keys()),
        index=index,
        key="patient_selector_box"
    )
    
    # 5. Update session state
    if selected_label:
        st.session_state['current_patient'] = patient_options[selected_label]
        
        # Display mini info
        p = patient_options[selected_label]
        st.sidebar.info(f"**Age:** {p.get('age')} | **Gender:** {p.get('gender')}")
