import streamlit as st
import requests

def show_risk_assessment(api_base: str):
    st.header("🔍 Patient Risk Assessment")
    
    current_patient = st.session_state.get('current_patient')
    
    if not current_patient:
        st.warning("Please select a patient from the sidebar first.")
        patient_id = st.text_input("Or enter Patient ID manually")
    else:
        st.success(f"Analysing Risk for: **{current_patient['name']}**")
        patient_id = current_patient['patient_id']

    st.markdown("### Configuration")
    risk_types = st.multiselect(
        "Select Risk Models",
        ["mortality", "readmission", "infection"],
        default=["mortality", "readmission"]
    )
    
    if st.button("Run Risk Assessment"):
        with st.spinner("Analyzing patient data..."):
            try:
                payload = {
                    "patient_id": patient_id,
                    "risk_types": risk_types
                }
                # If current patient is loaded, we could strictly rely on backend to fetch data 
                # OR send data if we allowed modification. For now, sending ID is cleaner.
                
                response = requests.post(f"{api_base}/ai-tools/risk-assessment", json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    res = data.get("risk_assessment", {})
                    summary = data.get("risk_summary", {})
                    
                    st.divider()
                    
                    # Overall
                    ov = summary.get("overall_risk_level", "Unknown")
                    color = "red" if ov == "High" else "orange" if ov == "Medium" else "green"
                    st.markdown(f"## Overall Risk Level: :{color}[{ov}]")
                    st.write(f"**Score:** {summary.get('overall_risk_score')}")
                    
                    st.subheader("Detailed Breakdown")
                    c1, c2, c3 = st.columns(3)
                    
                    if "mortality_risk" in res:
                        with c1:
                            r = res["mortality_risk"]
                            st.info(f"**Mortality**: {r.get('risk_level')}")
                            st.write(r.get('recommendations', []))

                    if "readmission_risk" in res:
                        with c2:
                            r = res["readmission_risk"]
                            st.info(f"**Readmission**: {r.get('risk_level')}")
                            st.write(r.get('recommendations', []))
                            
                    if "infection_risk" in res:
                        with c3:
                            r = res["infection_risk"]
                            st.info(f"**Infection**: {r.get('risk_level')}")
                            st.write(r.get('recommendations', []))
                            
                    st.success("Assessment Saved to Patient History.")
                    
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
