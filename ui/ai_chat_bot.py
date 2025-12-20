"""
HealthAI Assistant - Basic Chatbot Implementation
"""
import streamlit as st
import random
import time
import requests

def show_ai_chat_bot():
    """Display the AI Chatbot interface"""
    st.header("🤖 HealthAI Assistant")
    st.markdown("ask me anything about the system or the current patient.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your HealthAI Assistant. How can I help you today?"
        })

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # Get Current Patient ID
                    current_patient = st.session_state.get('current_patient')
                    pid = current_patient.get('patient_id') if current_patient else None
                    
                    # API Base (hack: get from sidebar input or default)
                    api_base = st.session_state.get('api_base_url_global', 'http://localhost:8000')

                    payload = {
                        "patient_id": pid,
                        "query": prompt,
                        "history": st.session_state.messages[-4:] # Send last few messages
                    }
                    
                    response = requests.post(f"{api_base}/ai-tools/chat", json=payload, timeout=30)
                    if response.status_code == 200:
                        full_response = response.json().get("response", "No response from AI.")
                    else:
                        full_response = f"Error: {response.text}"
                        
                except Exception as e:
                    full_response = f"Connection Failed: {e}"

            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
