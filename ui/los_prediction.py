import streamlit as st
import requests

def show_los_prediction(api_base: str):
    st.title("📈 LOS Prediction")
    st.subheader("Length of Stay Prediction")
    
    # 1. Access Patient Context
    patient = st.session_state.get('current_patient')
    
    # Defaults
    default_age = 45
    default_gender_ix = 0
    
    if patient:
        st.success(f"Context: {patient.get('name')} (ID: {patient.get('patient_id')})")
        val_age = patient.get('age', 45)
        try:
            default_age = int(val_age)
        except:
            default_age = 45
            
        val_gender = patient.get('gender', 'F')
        if val_gender.upper().startswith('M'):
            default_gender_ix = 1
        else:
            default_gender_ix = 0
    else:
        st.info("No patient selected. Using manual entry.")

    # 2. Enhanced Form
    with st.form("los_pred_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", value=default_age)
            admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
        with col2:
            gender = st.selectbox("Gender", ["F", "M"], index=default_gender_ix)
            num_diagnoses = st.number_input("Number of Diagnoses", value=1, min_value=1)
            
        st.markdown("**Vitals & Labs**")
        c1, c2, c3 = st.columns(3)
        with c1:
            bmi = st.number_input("BMI", value=25.0)
        with c2:
            wbc = st.number_input("White Blood Cells", value=8.0)
        with c3:
            creatinine = st.number_input("Creatinine", value=1.0)
            
        submit = st.form_submit_button("Predict LOS")
        
    # 3. Handle Submission
    if submit:
        # Mapping: Age, Gender, AdmType(encoded), NumDiag, BMI, WBC, Creatinine
        gender_num = 1 if gender == "M" else 0
        adm_map = {"Emergency": 1, "Urgent": 2, "Elective": 0}
        
        features = [
            float(age),
            float(gender_num),
            float(adm_map[admission_type]),
            float(num_diagnoses),
            float(bmi),
            float(wbc),
            float(creatinine)
        ]
        
        try:
            res = requests.post(f"{api_base}/predict/regress", json={"features": features}, timeout=10)
            if res.status_code == 200:
                result = res.json()
                pred = result.get('prediction', 0)
                st.markdown("### Results")
                st.metric("Predicted Length of Stay", f"{pred:.1f} days")
                st.caption(f"Model used: {result.get('model_type', 'unknown')}")
            else:
                st.error(f"Error: {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
