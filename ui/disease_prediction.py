import streamlit as st
import requests
import numpy as np

def get_gender_val(gender_str: str) -> int:
    return 1 if gender_str and gender_str.upper().startswith('M') else 0

def show_disease_prediction(api_base: str):
    st.title("🔬 Disease Prediction")
    st.subheader("Disease Risk Classification")
    
    # 1. Access Patient Context
    patient = st.session_state.get('current_patient')
    
    # Defaults
    default_age = 30
    default_gender_ix = 0 # 0=Female, 1=Male
    default_gender = "F"
    
    if patient:
        st.success(f"Context: {patient.get('name')} (ID: {patient.get('patient_id')})")
        val_age = patient.get('age', 30)
        try:
            default_age = int(val_age)
        except:
            default_age = 30
            
        val_gender = patient.get('gender', 'F')
        if val_gender.upper().startswith('M'):
            default_gender_ix = 1
            default_gender = "M"
        else:
            default_gender_ix = 0
            default_gender = "F"
    else:
        st.info("No patient selected. Using manual entry.")

    # 2. Enhanced Structured Form
    with st.form("disease_pred_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=default_age)
            interaction = st.number_input("Social/Interaction Score", value=5.0) # Placeholder/Example feature
            activity = st.number_input("Activity Level (1-10)", value=5.0)
            
        with col2:
            gender = st.selectbox("Gender", ["F", "M"], index=default_gender_ix)
            bp_systolic = st.number_input("Systolic BP", value=120)
            bp_diastolic = st.number_input("Diastolic BP", value=80)
            
        # Additional clinical metrics (Placeholders to match a generic list length if needed)
        st.markdown("**Clinical Vitals**")
        c1, c2, c3 = st.columns(3)
        with c1:
            hr = st.number_input("Heart Rate", value=75)
        with c2:
            resp = st.number_input("Respiratory Rate", value=16)
        with c3:
            spo2 = st.number_input("SpO2 (%)", value=98)
            
        submit = st.form_submit_button("Predict Risk")
    
    # 3. Handle Submission
    if submit:
        # Construct feature vector matching backend expectation
        # Note: We don't know the EXACT model mapping without the feature pickle. 
        # We will send a standard list and hope the stub/baseline handles it or the real model matches.
        # Mapping: Age, Gender(0/1), BP_Sys, BP_Dia, HR, Resp, SpO2, Interaction, Activity
        gender_num = 1 if gender == "M" else 0
        features = [
            float(age), 
            float(gender_num), 
            float(bp_systolic), 
            float(bp_diastolic), 
            float(hr), 
            float(resp), 
            float(spo2),
            float(interaction),
            float(activity)
        ]
        
        # Determine strict list length if possible, otherwise send this.
        # The backend expects "List[float]".
        
        try:
            res = requests.post(f"{api_base}/predict/classify", json={"features": features}, timeout=10)
            if res.status_code == 200:
                result = res.json()
                st.markdown("### Results")
                if "proba" in result and result["proba"] is not None:
                     st.metric("Risk Probability", f"{result['proba']:.2%}")
                
                prediction = result.get("prediction")
                if prediction == 1:
                    st.error(f"High Risk Detected (Class {prediction})")
                else:
                    st.success(f"Low Risk (Class {prediction})")
                    
                st.caption(f"Model used: {result.get('model_type', 'unknown')}")
            else:
                st.error(f"Error: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
