import streamlit as st
import requests

def show_image_diagnostics(api_base: str):
    st.header("🖼️ Image Diagnostics")
    
    current_patient = st.session_state.get('current_patient')
    patient_id = current_patient['patient_id'] if current_patient else None

    uploaded_file = st.file_uploader("Upload Medical Image (X-Ray, MRI)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file and st.button("Analyze Image"):
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        with st.spinner("Analyzing image patterns..."):
            try:
                # In a real app, send actual file. Here just triggering mock.
                payload = {"patient_id": patient_id, "image_type": "X-Ray"}
                response = requests.post(f"{api_base}/ai-tools/analyze-image", json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json().get("result", {})
                    if result:
                        st.success("Analysis Complete")
                        
                        if 'report_text' in result:
                            st.markdown("### 📋 AI-Generated Radiology Report")
                            st.markdown(result['report_text'])
                        else:
                            st.write(result)
                    else:
                        st.error("Analysis failed: No result returned from AI.")
                else:
                    st.error("Analysis failed.")
            except Exception as e:
                st.error(f"Error: {e}")
