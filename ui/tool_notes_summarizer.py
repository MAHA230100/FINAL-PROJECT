import streamlit as st
import requests

def show_notes_summarizer(api_base: str):
    st.header("📝 Notes Summarizer")
    
    current_patient = st.session_state.get('current_patient')
    patient_id = current_patient['patient_id'] if current_patient else None
    
    if current_patient:
        st.info(f"Patient: **{current_patient['name']}**")
    
    # Auto-load notes from DB if available
    current_patient = st.session_state.get('current_patient')
    default_text = ""
    if current_patient and 'clinical_notes' in current_patient:
        notes_list = current_patient['clinical_notes']
        if isinstance(notes_list, list):
            default_text = "\n\n".join([f"[{n.get('date')}] {n.get('content')}" for n in notes_list])
    
    notes_input = st.text_area("Clinical Notes (Auto-loaded from DB)", value=default_text, height=200,
                        placeholder="Patient complains of chest pain... History of hypertension...")
    
    if st.button("Summarize Notes"):
        if not notes_input:
            st.error("Please enter some notes.")
            return
            
        with st.spinner("Processing NLP..."):
            try:
                payload = {"notes_text": notes_input, "patient_id": patient_id}
                response = requests.post(f"{api_base}/ai-tools/summarize-notes", json=payload, timeout=30)
                if response.status_code == 200:
                    res = response.json().get("result", {})
                    st.subheader("Summary")
                    st.write(res.get("summary"))
                    
                    st.subheader("Key Findings")
                    for k in res.get("key_findings", []):
                        st.markdown(f"- {k}")
                else:
                    st.error("Failed to summarize.")
            except Exception as e:
                st.error(f"Error: {e}")
