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
        # Progressive loading with status updates
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Step 1: Fetching patient data
            status_container.info("🔍 Fetching patient data from database...")
            progress_bar.progress(20)
            import time
            time.sleep(0.5)
            
            payload = {
                "patient_id": patient_id,
                "risk_types": risk_types
            }
            
            # Step 2: Loading AI model
            status_container.info("🤖 Loading AI risk assessment model...")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            # Step 3: Processing analysis
            status_container.info("⚙️ Processing comprehensive risk analysis...")
            progress_bar.progress(60)
            
            response = requests.post(f"{api_base}/ai-tools/risk-assessment", json=payload, timeout=45)
            
            # Step 4: Generating report
            status_container.info("📊 Generating detailed risk report...")
            progress_bar.progress(80)
            time.sleep(0.3)
            
            if response.status_code == 200:
                data = response.json()
                
                # Complete
                progress_bar.progress(100)
                status_container.success("✅ Analysis complete!")
                time.sleep(0.5)
                status_container.empty()
                progress_bar.empty()
                
                st.divider()
                
                # Display overall risk level
                risk_level = data.get("overall_risk_level", "Unknown")
                risk_score = data.get("overall_risk_score", 0)
                
                color = "red" if "High" in risk_level else "orange" if "Medium" in risk_level else "green"
                st.markdown(f"## Overall Risk Level: :{color}[{risk_level}]")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Score", f"{risk_score}/100")
                with col2:
                    factors = data.get("factors", [])
                    st.metric("Risk Factors Identified", len(factors))
                
                # Display patient snapshot
                if 'patient_snapshot' in data:
                    with st.expander("📋 Patient Snapshot", expanded=False):
                        snapshot = data['patient_snapshot']
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Age", f"{snapshot.get('age')} years")
                            st.metric("Previous Admissions", snapshot.get('previous_admissions', 0))
                        with col_b:
                            st.metric("Blood Pressure", f"{snapshot.get('vitals_bp')} mmHg")
                            st.metric("Heart Rate", f"{snapshot.get('vitals_hr')} bpm")
                        with col_c:
                            st.metric("Comorbidities", snapshot.get('comorbidity_count', 0))
                
                # Display AI Analysis prominently
                if 'analysis' in data and data['analysis']:
                    st.markdown("### 🤖 Comprehensive AI Analysis")
                    # Check if it's a mock response
                    if "[MOCK]" in data['analysis']:
                        st.warning("⚠️ AI is running in mock mode. Real AI analysis requires valid GOOGLE_API_KEY in environment.")
                    st.markdown(data['analysis'])
                
                # Display detailed breakdown
                st.subheader("Detailed Breakdown by Risk Type")
                risk_assessment = data.get("risk_assessment", {})
                
                if risk_assessment:
                    cols = st.columns(min(len(risk_assessment), 3))
                    for idx, (risk_type, risk_data) in enumerate(risk_assessment.items()):
                        with cols[idx % 3]:
                            if isinstance(risk_data, dict):
                                level = risk_data.get('risk_level', 'Unknown')
                                score = risk_data.get('score', 0)
                                st.info(f"**{risk_type.replace('_', ' ').title()}**")
                                st.write(f"Level: {level}")
                                st.write(f"Score: {score}/100")
                            else:
                                st.info(f"**{risk_type.replace('_', ' ').title()}**: {risk_data}")
                
                st.success("✅ Assessment completed and saved to patient history.")
                
            else:
                progress_bar.empty()
                status_container.empty()
                st.error(f"Error: {response.text}")
        except Exception as e:
            progress_bar.empty()
            status_container.empty()
            st.error(f"Connection failed: {e}")
