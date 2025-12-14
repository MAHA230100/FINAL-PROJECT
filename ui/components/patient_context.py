import streamlit as st
import requests
import os

def patient_context(api_base_default="http://localhost:8000"):
    """Universal patient selection and loader (puts into st.session_state)"""
    api_base = st.sidebar.text_input("API base URL", api_base_default, key="api_base_url_global")
    patient_id = st.sidebar.text_input("Patient ID", value=st.session_state.get("patient_id", ""), key="sidebar_patient_id")
    load_patient = st.sidebar.button("Load Patient", key="btn_load_patient")
    if load_patient or (patient_id and "patient" not in st.session_state):
        try:
            response = requests.get(f"{api_base}/patients/{patient_id}", timeout=10)
            if response.status_code == 200:
                st.session_state["patient_id"] = patient_id
                st.session_state["patient"] = response.json()
                st.success(f"Loaded patient {patient_id}")
            else:
                st.session_state.pop("patient", None)
                st.error(f"Patient not found for ID: {patient_id}")
        except Exception as e:
            st.session_state.pop("patient", None)
            st.error(f"Failed to fetch patient: {e}")
    return api_base, st.session_state.get("patient_id"), st.session_state.get("patient")
