"""
AI Tools Demo Page - Showcase healthcare AI utilities
"""

import streamlit as st
import requests
import pandas as pd
import json
import os
from typing import Dict, Any

def show_ai_tools_demo():
    """AI Tools Demo page"""
    st.header("🤖 AI Tools Demo")
    st.write("Explore advanced healthcare AI utilities for patient analysis, risk assessment, and clinical guidance.")
    
    _default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_BASE = st.sidebar.text_input("API base URL", _default_api, key="api_base_url_ai_tools")
    
    # Check AI tools health
    try:
        health_response = requests.get(f"{API_BASE}/ai-tools/health-check", timeout=10)
        if health_response.status_code == 200:
            st.success("✅ AI Tools are healthy and ready!")
        else:
            st.warning("⚠️ AI Tools may not be fully operational")
    except Exception as e:
        st.error(f"❌ Cannot connect to AI Tools: {e}")
        return
    
    # Sample patient data
    sample_patient_data = {
        "patient_id": "P001",
        "age": 65,
        "gender": "Male",
        "comorbidities": "Hypertension, Diabetes, Heart Disease",
        "vitals_bp": 150,
        "vitals_hr": 85,
        "temperature": 98.6,
        "oxygen_saturation": 96,
        "respiratory_rate": 18,
        "medications": ["Lisinopril", "Metformin", "Aspirin", "Atorvastatin"],
        "diagnosis": "Heart Failure",
        "admission_date": "2024-01-01",
        "length_of_stay": 5,
        "discharge_disposition": "Home",
        "follow_up_scheduled": True,
        "icu_stay": False,
        "isolation": False,
        "ventilator": False,
        "immunocompromised": False,
        "cognitive_impairment": False,
        "caregiver_support": True,
        "lab_values": {
            "creatinine": 1.2,
            "glucose": 140,
            "wbc": 8000
        }
    }
    
    # AI Tools Selection
    st.subheader("🔧 Available AI Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Core AI Tools**")
        health_analysis = st.button("🏥 Health Analysis", key="health_analysis_btn")
        clinical_guidance = st.button("👨‍⚕️ Clinical Guidance", key="clinical_guidance_btn")
        risk_assessment = st.button("⚠️ Risk Assessment", key="risk_assessment_btn")
    
    with col2:
        st.write("**Specialized Tools**")
        medication_analysis = st.button("💊 Medication Analysis", key="medication_analysis_btn")
        patient_monitoring = st.button("📊 Patient Monitoring", key="patient_monitoring_btn")
        comprehensive_analysis = st.button("🔍 Comprehensive Analysis", key="comprehensive_analysis_btn")
    
    # Health Analysis
    if health_analysis:
        st.subheader("🏥 Health Analysis Results")
        try:
            response = requests.post(
                f"{API_BASE}/ai-tools/health-analysis",
                json={"patient_data": sample_patient_data},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                st.success("Health analysis completed!")
                st.json(result)
            else:
                st.error(f"Health analysis failed: {response.json().get('detail', response.text)}")
        except Exception as e:
            st.error(f"Error during health analysis: {e}")
    
    # Clinical Guidance
    if clinical_guidance:
        st.subheader("👨‍⚕️ Clinical Guidance Results")
        try:
            response = requests.post(
                f"{API_BASE}/ai-tools/clinical-guidance",
                json={"patient_data": sample_patient_data, "consultation_type": "general"},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                st.success("Clinical guidance generated!")
                st.json(result)
            else:
                st.error(f"Clinical guidance failed: {response.json().get('detail', response.text)}")
        except Exception as e:
            st.error(f"Error during clinical guidance: {e}")
    
    # Risk Assessment
    if risk_assessment:
        st.subheader("⚠️ Risk Assessment Results")
        try:
            response = requests.post(
                f"{API_BASE}/ai-tools/risk-assessment",
                json={
                    "patient_data": sample_patient_data,
                    "risk_types": ["mortality", "readmission", "infection"]
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                st.success("Risk assessment completed!")
                st.json(result)
            else:
                st.error(f"Risk assessment failed: {response.json().get('detail', response.text)}")
        except Exception as e:
            st.error(f"Error during risk assessment: {e}")
    
    # Medication Analysis
    if medication_analysis:
        st.subheader("💊 Medication Analysis Results")
        try:
            response = requests.post(
                f"{API_BASE}/ai-tools/medication-analysis",
                json={"patient_data": sample_patient_data, "analysis_type": "comprehensive"},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                st.success("Medication analysis completed!")
                st.json(result)
            else:
                st.error(f"Medication analysis failed: {response.json().get('detail', response.text)}")
        except Exception as e:
            st.error(f"Error during medication analysis: {e}")
    
    # Patient Monitoring
    if patient_monitoring:
        st.subheader("📊 Patient Monitoring Results")
        try:
            response = requests.post(
                f"{API_BASE}/ai-tools/patient-monitoring",
                json={"patient_data": sample_patient_data, "monitoring_type": "comprehensive"},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                st.success("Patient monitoring completed!")
                st.json(result)
            else:
                st.error(f"Patient monitoring failed: {response.json().get('detail', response.text)}")
        except Exception as e:
            st.error(f"Error during patient monitoring: {e}")
    
    # Comprehensive Analysis
    if comprehensive_analysis:
        st.subheader("🔍 Comprehensive AI Analysis")
        st.write("Running all AI tools for comprehensive patient analysis...")
        
        # Run all AI tools
        results = {}
        
        # Health Analysis
        try:
            health_response = requests.post(
                f"{API_BASE}/ai-tools/health-analysis",
                json={"patient_data": sample_patient_data},
                timeout=30
            )
            if health_response.status_code == 200:
                results["health_analysis"] = health_response.json()
        except Exception as e:
            st.error(f"Health analysis error: {e}")
        
        # Clinical Guidance
        try:
            clinical_response = requests.post(
                f"{API_BASE}/ai-tools/clinical-guidance",
                json={"patient_data": sample_patient_data, "consultation_type": "general"},
                timeout=30
            )
            if clinical_response.status_code == 200:
                results["clinical_guidance"] = clinical_response.json()
        except Exception as e:
            st.error(f"Clinical guidance error: {e}")
        
        # Risk Assessment
        try:
            risk_response = requests.post(
                f"{API_BASE}/ai-tools/risk-assessment",
                json={
                    "patient_data": sample_patient_data,
                    "risk_types": ["mortality", "readmission", "infection"]
                },
                timeout=30
            )
            if risk_response.status_code == 200:
                results["risk_assessment"] = risk_response.json()
        except Exception as e:
            st.error(f"Risk assessment error: {e}")
        
        # Medication Analysis
        try:
            medication_response = requests.post(
                f"{API_BASE}/ai-tools/medication-analysis",
                json={"patient_data": sample_patient_data, "analysis_type": "comprehensive"},
                timeout=30
            )
            if medication_response.status_code == 200:
                results["medication_analysis"] = medication_response.json()
        except Exception as e:
            st.error(f"Medication analysis error: {e}")
        
        # Patient Monitoring
        try:
            monitoring_response = requests.post(
                f"{API_BASE}/ai-tools/patient-monitoring",
                json={"patient_data": sample_patient_data, "monitoring_type": "comprehensive"},
                timeout=30
            )
            if monitoring_response.status_code == 200:
                results["patient_monitoring"] = monitoring_response.json()
        except Exception as e:
            st.error(f"Patient monitoring error: {e}")
        
        # Display results
        if results:
            st.success("Comprehensive analysis completed!")
            st.json(results)
        else:
            st.error("No AI tools were able to complete successfully")
    
    # Custom Patient Data Input
    st.markdown("---")
    st.subheader("🔧 Custom Patient Analysis")
    
    st.write("Enter custom patient data for analysis:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        custom_patient_id = st.text_input("Patient ID", value="P002", key="custom_patient_id")
        custom_age = st.number_input("Age", min_value=0, max_value=120, value=45, key="custom_age")
        custom_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="custom_gender")
        custom_comorbidities = st.text_input("Comorbidities", value="Hypertension", key="custom_comorbidities")
    
    with col2:
        custom_bp = st.number_input("Blood Pressure", min_value=50, max_value=250, value=140, key="custom_bp")
        custom_hr = st.number_input("Heart Rate", min_value=30, max_value=200, value=80, key="custom_hr")
        custom_temp = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6, key="custom_temp")
        custom_o2 = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=98, key="custom_o2")
    
    custom_medications = st.text_input("Medications (comma-separated)", value="Lisinopril, Metformin", key="custom_medications")
    custom_diagnosis = st.text_input("Diagnosis", value="Hypertension", key="custom_diagnosis")
    
    if st.button("Analyze Custom Patient", key="analyze_custom_patient"):
        custom_patient_data = {
            "patient_id": custom_patient_id,
            "age": custom_age,
            "gender": custom_gender,
            "comorbidities": custom_comorbidities,
            "vitals_bp": custom_bp,
            "vitals_hr": custom_hr,
            "temperature": custom_temp,
            "oxygen_saturation": custom_o2,
            "medications": [med.strip() for med in custom_medications.split(",")],
            "diagnosis": custom_diagnosis,
            "admission_date": "2024-01-01",
            "length_of_stay": 3,
            "discharge_disposition": "Home",
            "follow_up_scheduled": True,
            "icu_stay": False,
            "isolation": False,
            "ventilator": False,
            "immunocompromised": False,
            "cognitive_impairment": False,
            "caregiver_support": True,
            "lab_values": {
                "creatinine": 1.0,
                "glucose": 120,
                "wbc": 7000
            }
        }
        
        # Run comprehensive analysis on custom patient
        st.write("Running comprehensive analysis on custom patient...")
        
        try:
            # Health Analysis
            health_response = requests.post(
                f"{API_BASE}/ai-tools/health-analysis",
                json={"patient_data": custom_patient_data},
                timeout=30
            )
            if health_response.status_code == 200:
                st.success("✅ Health analysis completed for custom patient!")
                st.json(health_response.json())
            else:
                st.error(f"Health analysis failed: {health_response.json().get('detail', health_response.text)}")
        except Exception as e:
            st.error(f"Error during custom patient analysis: {e}")
    
    # AI Tools Information
    st.markdown("---")
    st.subheader("ℹ️ AI Tools Information")
    
    try:
        tools_response = requests.get(f"{API_BASE}/ai-tools/tools", timeout=10)
        if tools_response.status_code == 200:
            tools_info = tools_response.json()
            st.write("**Available AI Tools:**")
            for tool in tools_info.get("available_tools", []):
                st.write(f"• **{tool['name']}**: {tool['description']}")
        else:
            st.warning("Could not retrieve AI tools information")
    except Exception as e:
        st.warning(f"Error retrieving AI tools information: {e}")
    
    # Usage Instructions
    st.markdown("---")
    st.subheader("📖 Usage Instructions")
    
    st.write("""
    **How to use the AI Tools:**
    
    1. **Health Analysis**: Provides AI-powered health insights and risk assessment
    2. **Clinical Guidance**: Offers clinical recommendations and treatment plans
    3. **Risk Assessment**: Evaluates mortality, readmission, and infection risks
    4. **Medication Analysis**: Analyzes medications for interactions and optimization
    5. **Patient Monitoring**: Provides continuous monitoring and alerting capabilities
    
    **Features:**
    - Real-time AI analysis
    - Comprehensive risk assessment
    - Medication optimization
    - Clinical decision support
    - Patient monitoring and alerting
    
    **Note**: These are demonstration tools with mock AI capabilities. In a production environment, 
    they would integrate with real AI models and healthcare data systems.
    """)