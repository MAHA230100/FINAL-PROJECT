import streamlit as st
import requests
import os

def show_ai_tools_demo():
    """AI Tools demonstration page"""
    st.header("🛠️ AI Tools Demo")
    
    _default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_BASE = st.sidebar.text_input("API base URL", _default_api, key="api_base_url_ai")
    
    # Available AI Tools
    st.subheader("Available AI Tools")
    
    if st.button("Load Available Tools", key="load_tools"):
        try:
            response = requests.get(f"{API_BASE}/ai-tools/available", timeout=10)
            tools = response.json()
            
            if "tools" in tools:
                for tool in tools["tools"]:
                    st.write(f"- **{tool['name']}**: {tool['description']}")
        except Exception as e:
            st.error(f"Failed to load tools: {e}")
    
    # Text Summarization
    st.subheader("📝 Text Summarization")
    
    text_input = st.text_area(
        "Enter text to summarize:",
        value="This is a sample text that will be summarized by our AI tool. The summarization feature can help extract key information from long documents and provide concise summaries.",
        height=100,
        key="summarize_text"
    )
    
    max_length = st.slider("Maximum summary length", 50, 500, 150, key="max_length")
    
    if st.button("Summarize Text", key="summarize"):
        if text_input.strip():
            try:
                response = requests.post(
                    f"{API_BASE}/ai-tools/summarize",
                    json={"text": text_input, "max_length": max_length},
                    timeout=30
                )
                result = response.json()
                
                st.success("Text summarized successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Summary:**")
                    st.write(result.get("summary", "No summary available"))
                
                with col2:
                    st.write("**Keywords:**")
                    keywords = result.get("keywords", [])
                    if keywords:
                        st.write(", ".join(keywords))
                    else:
                        st.write("No keywords extracted")
                
                # Show statistics
                st.write("**Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", result.get("original_length", 0))
                with col2:
                    st.metric("Summary Length", result.get("summary_length", 0))
                with col3:
                    compression_ratio = result.get("original_length", 1) / max(result.get("summary_length", 1), 1)
                    st.metric("Compression Ratio", f"{compression_ratio:.1f}x")
                
            except Exception as e:
                st.error(f"Summarization failed: {e}")
        else:
            st.warning("Please enter some text to summarize")
    
    # Text Classification
    st.subheader("🏷️ Text Classification")
    
    classify_text = st.text_area(
        "Enter text to classify:",
        value="I love this product! It works perfectly and exceeded my expectations.",
        height=100,
        key="classify_text"
    )
    
    if st.button("Classify Text", key="classify"):
        if classify_text.strip():
            try:
                response = requests.post(
                    f"{API_BASE}/ai-tools/classify",
                    json={"text": classify_text},
                    timeout=30
                )
                result = response.json()
                
                st.success("Text classified successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Classification Result:**")
                    label = result.get("label", "Unknown")
                    confidence = result.get("confidence", 0)
                    st.write(f"**Label:** {label}")
                    st.write(f"**Confidence:** {confidence:.3f}")
                
                with col2:
                    # Show confidence as a progress bar
                    st.write("**Confidence Level:**")
                    st.progress(confidence)
                
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.warning("Please enter some text to classify")
    
    # Future AI Tools (placeholders)
    st.subheader("🚀 Coming Soon")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Text Translation**\n\nTranslate medical text between languages using advanced NLP models.")
    
    with col2:
        st.info("**Sentiment Analysis**\n\nAnalyze patient feedback sentiment for quality improvement.")
