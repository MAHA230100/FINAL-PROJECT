import streamlit as st
import requests

def show_feedback_analysis(api_base: str):
    st.header("💬 Patient Feedback Analysis")
    
    current_patient = st.session_state.get('current_patient')
    patient_id = current_patient['patient_id'] if current_patient else None
    
    feedback = st.text_area("Patient Feedback / Complaint", height=150)
    
    if st.button("Analyze Sentiment"):
        if not feedback:
            st.error("Enter feedback text.")
            return
            
        with st.spinner("Analyzing sentiment..."):
            try:
                payload = {"feedback_text": feedback, "patient_id": patient_id}
                response = requests.post(f"{api_base}/ai-tools/analyze-feedback", json=payload, timeout=30)
                if response.status_code == 200:
                    res = response.json().get("result", {})
                    st.metric("Sentiment", res.get("sentiment"))
                    st.write(f"**Score:** {res.get('score')}")
                    st.write("### Topics Detected")
                    st.write(res.get("topics"))
                else:
                    st.error("Failed.")
            except Exception as e:
                st.error(f"Error: {e}")
