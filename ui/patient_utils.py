import streamlit as st
import requests
import os

def get_patient(api_base, patient_id):
    """Fetch patient data via API and cache in session, returns dict or None."""
    cache_key = f"patient_{patient_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        response = requests.get(f"{api_base}/patients/{patient_id}", timeout=10)
        if response.status_code == 200:
            patient = response.json()
            st.session_state[cache_key] = patient
            return patient
    except Exception:
        pass
    return None
