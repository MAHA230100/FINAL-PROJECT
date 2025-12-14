"""
AI Tools API Routes - Healthcare AI utilities endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json

from ..ai_tools import (
    HealthAnalyzer, 
    ClinicalAdvisor, 
    RiskAssessor, 
    MedicationAdvisor, 
    PatientMonitor
)
from .hospital_db_service import fetch_patient, fetch_patient_comprehensive, save_prediction

router = APIRouter(prefix="/ai-tools", tags=["ai-tools"])

# Initialize AI tools
health_analyzer = HealthAnalyzer()
clinical_advisor = ClinicalAdvisor()
risk_assessor = RiskAssessor()
medication_advisor = MedicationAdvisor()
patient_monitor = PatientMonitor()

# Helper to enrich partial patient data
def _enrich_patient_data(patient_id: Optional[str], provided_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    final_data = {}
    
    # 1. Fetch Key Data
    if patient_id:
        db_patient = fetch_patient_comprehensive(patient_id)
        if db_patient:
            final_data.update(db_patient)
            
    # 2. Merge Provided Data (override DB if specific)
    if provided_data:
        final_data.update(provided_data)
        
    # 3. Add Mock Vitals/Labs if missing (Bridge until Checkpoint 5)
    # This prevents AI tools from crashing on "fresh" patients that only have Name/Age
    defaults = {
        "vitals_bp": 120,
        "vitals_hr": 75,
        "temperature": 98.6,
        "oxygen_saturation": 98,
        "respiratory_rate": 16,
        "medications": [],
        "lab_values": {"creatinine": 1.0, "glucose": 100, "wbc": 7000},
        "comorbidities": "None Known"
    }
    for k, v in defaults.items():
        if k not in final_data:
            final_data[k] = v
            
    return final_data

# Request/Response Models
class HealthAnalysisRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_data: Optional[Dict[str, Any]] = None

class ClinicalGuidanceRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_data: Optional[Dict[str, Any]] = None
    consultation_type: str = "general"

class RiskAssessmentRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_data: Optional[Dict[str, Any]] = None
    risk_types: List[str] = ["mortality", "readmission", "infection"]

class MedicationAnalysisRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_data: Optional[Dict[str, Any]] = None
    analysis_type: str = "comprehensive"

class PatientMonitoringRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_data: Optional[Dict[str, Any]] = None
    monitoring_type: str = "comprehensive"

# Health Analysis Endpoints
@router.post("/health-analysis")
def analyze_health(req: HealthAnalysisRequest):
    """Perform AI-powered health analysis"""
    try:
        p_data = _enrich_patient_data(req.patient_id, req.patient_data)
        result = health_analyzer.analyze_patient_data(p_data)
        
        # Save result
        if p_data.get('patient_id'):
            save_prediction(p_data['patient_id'], "Health Analyzer", result)
            
        return {
            "status": "success",
            "analysis": result,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health analysis failed: {e}")

@router.post("/clinical-guidance")
def get_clinical_guidance(req: ClinicalGuidanceRequest):
    """Get AI-powered clinical guidance"""
    try:
        p_data = _enrich_patient_data(req.patient_id, req.patient_data)
        
        # Analyze vital signs
        vitals = {
            'blood_pressure': p_data.get('vitals_bp', 0),
            'heart_rate': p_data.get('vitals_hr', 0),
            'temperature': p_data.get('temperature', 98.6),
            'oxygen_saturation': p_data.get('oxygen_saturation', 98)
        }
        
        vital_analysis = clinical_advisor.analyze_vital_signs(vitals)
        treatment_plan = clinical_advisor.recommend_treatment_plan(p_data)
        drug_interactions = clinical_advisor.assess_drug_interactions(p_data.get('medications', []))
        discharge_summary = clinical_advisor.generate_discharge_summary(p_data)
        clinical_guidance = clinical_advisor.provide_clinical_guidance(p_data)
        
        result_pkg = {
            "vital_analysis": vital_analysis,
            "treatment_plan": treatment_plan,
            "drug_interactions": drug_interactions,
            "discharge_summary": discharge_summary,
            "clinical_guidance": clinical_guidance
        }
        
        # Save result
        if p_data.get('patient_id'):
            save_prediction(p_data['patient_id'], "Clinical Advisor", result_pkg)
        
        return {
            "status": "success",
            "clinical_guidance": result_pkg,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clinical guidance failed: {e}")

@router.post("/risk-assessment")
def assess_risk(req: RiskAssessmentRequest):
    """Perform comprehensive risk assessment"""
    try:
        p_data = _enrich_patient_data(req.patient_id, req.patient_data)
        risk_results = {}
        
        if "mortality" in req.risk_types:
            risk_results["mortality_risk"] = risk_assessor.calculate_mortality_risk(p_data)
        
        if "readmission" in req.risk_types:
            risk_results["readmission_risk"] = risk_assessor.calculate_readmission_risk(p_data)
        
        if "infection" in req.risk_types:
            risk_results["infection_risk"] = risk_assessor.calculate_infection_risk(p_data)
        
        # Generate overall risk summary
        risk_summary = risk_assessor.generate_risk_summary(p_data)
        
        result_pkg = {
            "risk_assessment": risk_results,
            "risk_summary": risk_summary
        }
        
        # Save result
        if p_data.get('patient_id'):
            save_prediction(p_data['patient_id'], "Risk Assessor", result_pkg)
        
        return {
            "status": "success",
            **result_pkg,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {e}")

@router.post("/medication-analysis")
def analyze_medications(req: MedicationAnalysisRequest):
    """Analyze medications and provide recommendations"""
    try:
        p_data = _enrich_patient_data(req.patient_id, req.patient_data)
        medications = p_data.get('medications', [])
        
        if req.analysis_type == "comprehensive":
            # Comprehensive medication analysis
            medication_analysis = medication_advisor.analyze_medication_list(medications, p_data)
            change_recommendations = medication_advisor.recommend_medication_changes(medications, p_data)
            optimized_regimen = medication_advisor.optimize_medication_regimen(p_data)
            adherence_assessment = medication_advisor.check_medication_adherence(p_data)
            medication_summary = medication_advisor.generate_medication_summary(p_data)
            
            result_pkg = {
                "current_analysis": medication_analysis,
                "change_recommendations": change_recommendations,
                "optimized_regimen": optimized_regimen,
                "adherence_assessment": adherence_assessment,
                "medication_summary": medication_summary
            }
            
            if p_data.get('patient_id'):
                save_prediction(p_data['patient_id'], "Medication Advisor", result_pkg)
            
            return {
                "status": "success",
                "medication_analysis": result_pkg,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        else:
            # Basic analysis
            medication_analysis = medication_advisor.analyze_medication_list(medications, p_data)
            return {
                "status": "success",
                "medication_analysis": medication_analysis,
                "timestamp": "2024-01-01T00:00:00Z"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medication analysis failed: {e}")

@router.post("/patient-monitoring")
def monitor_patient(req: PatientMonitoringRequest):
    """Monitor patient and generate alerts"""
    try:
        p_data = _enrich_patient_data(req.patient_id, req.patient_data)
        vitals = {
            'blood_pressure': p_data.get('vitals_bp', 0),
            'heart_rate': p_data.get('vitals_hr', 0),
            'temperature': p_data.get('temperature', 98.6),
            'oxygen_saturation': p_data.get('oxygen_saturation', 98),
            'respiratory_rate': p_data.get('respiratory_rate', 16)
        }
        
        if req.monitoring_type == "comprehensive":
            # Comprehensive monitoring
            vital_analysis = patient_monitor.analyze_vital_signs(vitals, p_data)
            monitoring_plan = patient_monitor.generate_monitoring_plan(p_data)
            medication_effects = patient_monitor.check_medication_effects(p_data)
            patient_summary = patient_monitor.generate_patient_summary(p_data)
            
            result_pkg = {
                "vital_analysis": vital_analysis,
                "monitoring_plan": monitoring_plan,
                "medication_effects": medication_effects,
                "patient_summary": patient_summary
            }
            
            if p_data.get('patient_id'):
                save_prediction(p_data['patient_id'], "Patient Monitor", result_pkg)
            
            return {
                "status": "success",
                "patient_monitoring": result_pkg,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        else:
            # Basic monitoring
            vital_analysis = patient_monitor.analyze_vital_signs(vitals, p_data)
            return {
                "status": "success",
                "vital_analysis": vital_analysis,
                "timestamp": "2024-01-01T00:00:00Z"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patient monitoring failed: {e}")

# --- AI Tools (LLM Powered) ---
from ..services.llm_service import llm_service

class ChatRequest(BaseModel):
    patient_id: Optional[str] = None
    query: str
    history: Optional[List[Dict[str, str]]] = None

@router.post("/chat")
def chat_with_ai(req: ChatRequest):
    """Interactive chat with LLM context-aware of patient"""
    try:
        p_data = _enrich_patient_data(req.patient_id, None)
        
        # Add limited history context if provided
        context_str = ""
        if req.history:
            context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in req.history[-5:]])
        
        # Construct dynamic prompt
        prompt = (
            f"Context: Patient {p_data.get('name', 'Unknown')}. "
            f"Vitals: BP {p_data.get('vitals_bp')}, HR {p_data.get('vitals_hr')}. "
            f"Previous interaction:\n{context_str}\n"
            f"User Question: {req.query}"
        )
        
        response_text = llm_service.analyze_patient(p_data, f"Answer this user question as a clinical assistant: {req.query}")
        
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


class NotesSummaryRequest(BaseModel):
    patient_id: Optional[str] = None
    notes_text: str

@router.post("/summarize-notes")
def summarize_notes(req: NotesSummaryRequest):
    """Summarize clinical notes using LLM"""
    try:
        prompt = f"Summarize the following clinical notes and extract key medical terms:\n\n{req.notes_text}"
        summary = llm_service.generate_response(prompt)
        
        # Attempt to parse key findings if LLM returns text, or just split generic lines
        # For robustness, we'll just treat the whole text as meaningful
        
        result = {
            "summary": summary,
            "key_findings": ["See summary for details"],
            "sentiment": "Neutral" # Could ask LLM for this too
        }
        
        if req.patient_id:
             save_prediction(req.patient_id, "Notes Summarizer", result)

        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Notes summary failed: {e}")

class ImageAnalysisRequest(BaseModel):
    patient_id: Optional[str] = None
    image_type: str = "X-Ray"

@router.post("/analyze-image")
def analyze_image(req: ImageAnalysisRequest):
    """Analyze medical imagery"""
    try:
        # Mock Image Analysis (LLM Vision not implemented in this snippet yet, requires image bytes)
        result = {
            "finding": "No acute abnormalities detected (Simulated)",
            "confidence": 0.98,
            "regions_of_interest": ["Left Lower Lobe", "Cardiac Silhouette"]
        }
        
        if req.patient_id:
             save_prediction(req.patient_id, "Image Diagnostics", result)

        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {e}")

class FeedbackAnalysisRequest(BaseModel):
    patient_id: Optional[str] = None
    feedback_text: str

@router.post("/analyze-feedback")
def analyze_feedback(req: FeedbackAnalysisRequest):
    """Analyze patient feedback sentiment using LLM"""
    try:
        prompt = f"Analyze the sentiment of this patient feedback. Return a valid JSON with fields: sentiment (Positive/Negative/Neutral), score (0-1), and topics (list of strings).\n\nFeedback: {req.feedback_text}"
        
        analysis_text = llm_service.generate_response(prompt)
        
        # Simple/Naive parsing if LLM returns text instead of strict JSON
        # In prod, we'd use strict JSON mode or Pydantic parsers
        
        result = {
            "raw_analysis": analysis_text,
            "sentiment": "Mixed", # Fallback
            "score": 0.5,
            "topics": ["General Feedback"]
        }
        
        if req.patient_id:
             save_prediction(req.patient_id, "Feedback Analysis", result)

        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback analysis failed: {e}")

# Utility Endpoints
@router.get("/tools")
def list_ai_tools():
    """List available AI tools"""
    return {
        "available_tools": [
            {
                "name": "Health Analyzer",
                "description": "AI-powered health analysis and insights",
                "endpoint": "/ai-tools/health-analysis"
            },
            {
                "name": "Clinical Advisor",
                "description": "Clinical guidance and treatment recommendations",
                "endpoint": "/ai-tools/clinical-guidance"
            },
            {
                "name": "Risk Assessor",
                "description": "Comprehensive risk assessment (mortality, readmission, infection)",
                "endpoint": "/ai-tools/risk-assessment"
            },
            {
                "name": "Medication Advisor",
                "description": "Medication analysis and optimization",
                "endpoint": "/ai-tools/medication-analysis"
            },
            {
                "name": "Patient Monitor",
                "description": "Continuous patient monitoring and alerting",
                "endpoint": "/ai-tools/patient-monitoring"
            }
        ],
        "status": "success"
    }

@router.get("/health-check")
def ai_tools_health_check():
    """Health check for AI tools"""
    from ..services.llm_service import llm_service
    llm_status = "active" if llm_service.is_active() else "inactive (missing key)"
    
    return {
        "status": "healthy",
        "llm_status": llm_status,
        "tools_available": 5,
        "tools": [
            "HealthAnalyzer",
            "ClinicalAdvisor", 
            "RiskAssessor",
            "MedicationAdvisor",
            "PatientMonitor"
        ]
    }