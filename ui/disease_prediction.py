import streamlit as st
import requests
import numpy as np

def get_gender_val(gender_str: str) -> int:
    return 1 if gender_str and gender_str.upper().startswith('M') else 0

def show_disease_prediction(api_base: str):
    st.title("🔬 Disease Prediction")
    st.markdown("### comprehensive 14-Factor Analysis")
    
    # 1. Access Patient Context (Auto-fill)
    current_patient = st.session_state.get('current_patient')
    if current_patient is None:
        current_patient = {}
    
    # Defaults
    d_age = current_patient.get('age', 45)
    d_gender = current_patient.get('gender', 'Male')
    d_admit = current_patient.get('admission_type', 'Emergency')
    d_loc = current_patient.get('admission_location', 'Emergency Room')
    d_ins = current_patient.get('insurance', 'Medicare')
    d_lang = current_patient.get('language', 'English')
    d_mar = current_patient.get('marital_status', 'Married')
    d_drg = current_patient.get('drg_type', 'DRG-A')
    
    # Flatten complex lists for display defaults
    d_comorb = current_patient.get('comorbidities', [])
    if isinstance(d_comorb, list): d_comorb_str = ", ".join(d_comorb)
    else: d_comorb_str = str(d_comorb)

    d_bp = current_patient.get('vitals_bp', 120)
    d_hr = current_patient.get('vitals_hr', 75)
    d_prev = current_patient.get('previous_admissions', 0)
    d_risk = current_patient.get('risk_score', 0.5)

    if current_patient:
        st.success(f"Context Loaded: {current_patient.get('name')} (ID: {current_patient.get('patient_id')})")
    
    with st.form("disease_rich_form"):
        st.subheader("Demographics & Social")
        c1, c2, c3, c4 = st.columns(4)
        with c1: age = st.number_input("Age", value=int(d_age))
        with c2: gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if d_gender.upper().startswith('M') else 1)
        with c3: marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"], index=["Single", "Married", "Divorced", "Widowed"].index(d_mar) if d_mar in ["Single", "Married", "Divorced", "Widowed"] else 1)
        with c4: language = st.selectbox("Language", ["English", "Spanish", "Other"], index=0)

        st.subheader("Admission Details")
        c5, c6, c7, c8 = st.columns(4)
        with c5: admit_type = st.selectbox("Type", ["Emergency", "Urgent", "Elective"], index=["Emergency", "Urgent", "Elective"].index(d_admit) if d_admit in ["Emergency", "Urgent", "Elective"] else 0)
        with c6: admit_loc = st.selectbox("Location", ["Emergency Room", "Physician Referral", "Clinic Referral"], index=0)
        with c7: insurance = st.selectbox("Insurance", ["Medicare", "Medicaid", "Private", "Self-pay"], index=["Medicare", "Medicaid", "Private", "Self-pay"].index(d_ins) if d_ins in ["Medicare", "Medicaid", "Private", "Self-pay"] else 2)
        with c8: drg = st.text_input("DRG Code", value=str(d_drg))

        st.subheader("Clinical Factors")
        c9, c10 = st.columns(2)
        with c9: 
            comorbs = st.text_input("Comorbidities", value=d_comorb_str, help="Comma separated")
        with c10:
            lab_res = st.selectbox("Lab Results Summary", ["Normal", "Abnormal", "Critical"], index=1 if "Abnormal" in str(current_patient.get('lab_results')) else 0)
        
        st.subheader("Vitals & History")
        c11, c12, c13, c14 = st.columns(4)
        with c11: bp = st.number_input("Systolic BP", value=int(d_bp))
        with c12: hr = st.number_input("Heart Rate", value=int(d_hr))
        with c13: prev_adm = st.number_input("Previous Admissions", value=int(d_prev))
        with c14: r_score = st.slider("Calculated Risk Score", 0.0, 1.0, value=float(d_risk))
        
        submit = st.form_submit_button("Run Prediction (AI)")
        
    if submit:
        # Construct JSON payload
        payload = {
            "age": age,
            "gender": gender,
            "admission_type": admit_type,
            "admission_location": admit_loc,
            "insurance": insurance,
            "language": language,
            "marital_status": marital,
            "drg_type": drg,
            "comorbidities": comorbs,
            "lab_results": lab_res,
            "vitals_bp": bp,
            "vitals_hr": hr,
            "previous_admissions": prev_adm,
            "risk_score": r_score
        }
        
        try:
            res = requests.post(f"{api_base}/predict/classify", json=payload, timeout=15)
            if res.status_code == 200:
                data = res.json()
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    proba = data.get('proba', 0)
                    st.metric("Risk Probability", f"{proba:.1%}", delta_color="inverse")
                    if proba > 0.5:
                        st.error("High Risk of Readmission/Mortality")
                    else:
                        st.success("Low Risk")
                        
                with col_res2:
                    st.info(f"AI Reasoning: {data.get('reasoning', 'Analysis based on clinical features.')}")
                    
                with st.expander("Feature Data Sent"):
                    st.json(payload)
            else:
                st.error(f"Error: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
