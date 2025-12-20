import streamlit as st
import requests
import time

def show_feedback_analysis(api_base: str):
    st.header("💬 Patient Feedback Analysis")
    
    current_patient = st.session_state.get('current_patient')
    if current_patient is None:
        current_patient = {}
    
    patient_id = current_patient.get('patient_id')
    
    if current_patient and current_patient.get('name'):
        st.info(f"👤 Patient Context: **{current_patient.get('name')}** (ID: {patient_id})")
    else:
        st.info("No patient selected. Analysis will be generic.")
    
    feedback = st.text_area(
        "Patient Feedback / Complaint", 
        height=150,
        placeholder="Enter patient feedback, complaint, or review here..."
    )
    
    if st.button("🔍 Analyze Sentiment & Extract Insights"):
        if not feedback or len(feedback) < 5:
            st.error("Please enter feedback text (minimum 5 characters).")
            return
        
        # Progressive loading
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Step 1: Preparing
            status_container.info("📝 Preparing feedback analysis...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            # Step 2: Analyzing
            status_container.info("🤖 Running AI sentiment analysis...")
            progress_bar.progress(50)
            
            payload = {
                "feedback_text": feedback, 
                "patient_id": patient_id or ""
            }
            
            # Use new AI v2 endpoint
            response = requests.post(
                f"{api_base}/ai-v2/analyze-feedback", 
                json=payload, 
                timeout=30
            )
            
            # Step 3: Processing
            status_container.info("📊 Processing results...")
            progress_bar.progress(75)
            time.sleep(0.2)
            
            if response.status_code == 200:
                result = response.json()
                
                # Complete
                progress_bar.progress(100)
                status_container.success("✅ Analysis complete!")
                time.sleep(0.3)
                status_container.empty()
                progress_bar.empty()
                
                # Handle different response statuses
                status = result.get("status")
                
                if status == "missing_data":
                    st.warning(result.get("user_message", "Required data missing"))
                    return
                elif status == "error":
                    st.error(result.get("user_message", "Analysis failed"))
                    return
                
                # Success - display results
                data = result.get("data", {})
                
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                # Sentiment
                sentiment = data.get("sentiment", "neutral")
                sentiment_colors = {
                    "positive": "🟢",
                    "neutral": "🟡", 
                    "negative": "🔴"
                }
                sentiment_icon = sentiment_colors.get(sentiment, "⚪")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Sentiment", 
                        f"{sentiment_icon} {sentiment.title()}"
                    )
                with col2:
                    priority = data.get("priority", "medium")
                    st.metric("Priority", priority.title())
                with col3:
                    confidence = data.get("confidence", 0)
                    st.metric("Confidence", f"{confidence:.0%}")
                
                # Summary
                if data.get("summary"):
                    st.markdown("### 📝 Summary")
                    st.info(data.get("summary"))
                
                # Categories
                categories = data.get("categories", [])
                if categories:
                    st.markdown("### 🏷️ Categories Identified")
                    cols = st.columns(min(len(categories), 4))
                    for idx, cat in enumerate(categories):
                        with cols[idx % 4]:
                            st.button(cat, key=f"cat_{idx}", disabled=True)
                
                # Action Items
                action_items = data.get("action_items", [])
                if action_items:
                    st.markdown("### ✅ Recommended Actions")
                    for item in action_items:
                        priority_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                            item.get("priority", "medium"), "⚪"
                        )
                        with st.expander(f"{priority_icon} {item.get('category', 'Action')} - {item.get('priority', 'medium').title()} Priority"):
                            st.write(f"**Action:** {item.get('action', 'N/A')}")
                
                st.success("✅ Feedback analysis saved to patient record.")
                
            else:
                progress_bar.empty()
                status_container.empty()
                st.error(f"API Error: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.Timeout:
            progress_bar.empty()
            status_container.empty()
            st.error("⏱️ Request timed out. Please try again.")
        except Exception as e:
            progress_bar.empty()
            status_container.empty()
            st.error(f"❌ Error: {str(e)}")
