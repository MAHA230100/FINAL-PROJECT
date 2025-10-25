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

router = APIRouter(prefix="/ai-tools", tags=["ai-tools"])

# Initialize AI tools
health_analyzer = HealthAnalyzer()
clinical_advisor = ClinicalAdvisor()
risk_assessor = RiskAssessor()
medication_advisor = MedicationAdvisor()
patient_monitor = PatientMonitor()

# Request/Response Models
class HealthAnalysisRequest(BaseModel):
    patient_data: Dict[str, Any]

class ClinicalGuidanceRequest(BaseModel):
    patient_data: Dict[str, Any]
    consultation_type: str = "general"

class RiskAssessmentRequest(BaseModel):
    patient_data: Dict[str, Any]
    risk_types: List[str] = ["mortality", "readmission", "infection"]

class MedicationAnalysisRequest(BaseModel):
    patient_data: Dict[str, Any]
    analysis_type: str = "comprehensive"

class PatientMonitoringRequest(BaseModel):
    patient_data: Dict[str, Any]
    monitoring_type: str = "comprehensive"

# Health Analysis Endpoints
@router.post("/health-analysis")
def analyze_health(req: HealthAnalysisRequest):
    """Perform AI-powered health analysis"""
    try:
        result = health_analyzer.analyze_patient_data(req.patient_data)
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
        # Analyze vital signs
        vitals = {
            'blood_pressure': req.patient_data.get('vitals_bp', 0),
            'heart_rate': req.patient_data.get('vitals_hr', 0),
            'temperature': req.patient_data.get('temperature', 98.6),
            'oxygen_saturation': req.patient_data.get('oxygen_saturation', 98)
        }
        
        vital_analysis = clinical_advisor.analyze_vital_signs(vitals)
        treatment_plan = clinical_advisor.recommend_treatment_plan(req.patient_data)
        drug_interactions = clinical_advisor.assess_drug_interactions(req.patient_data.get('medications', []))
        discharge_summary = clinical_advisor.generate_discharge_summary(req.patient_data)
        clinical_guidance = clinical_advisor.provide_clinical_guidance(req.patient_data)
        
        return {
            "status": "success",
            "clinical_guidance": {
                "vital_analysis": vital_analysis,
                "treatment_plan": treatment_plan,
                "drug_interactions": drug_interactions,
                "discharge_summary": discharge_summary,
                "clinical_guidance": clinical_guidance
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clinical guidance failed: {e}")

@router.post("/risk-assessment")
def assess_risk(req: RiskAssessmentRequest):
    """Perform comprehensive risk assessment"""
    try:
        risk_results = {}
        
        if "mortality" in req.risk_types:
            risk_results["mortality_risk"] = risk_assessor.calculate_mortality_risk(req.patient_data)
        
        if "readmission" in req.risk_types:
            risk_results["readmission_risk"] = risk_assessor.calculate_readmission_risk(req.patient_data)
        
        if "infection" in req.risk_types:
            risk_results["infection_risk"] = risk_assessor.calculate_infection_risk(req.patient_data)
        
        # Generate overall risk summary
        risk_summary = risk_assessor.generate_risk_summary(req.patient_data)
        
        return {
            "status": "success",
            "risk_assessment": risk_results,
            "risk_summary": risk_summary,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {e}")

@router.post("/medication-analysis")
def analyze_medications(req: MedicationAnalysisRequest):
    """Analyze medications and provide recommendations"""
    try:
        medications = req.patient_data.get('medications', [])
        
        if req.analysis_type == "comprehensive":
            # Comprehensive medication analysis
            medication_analysis = medication_advisor.analyze_medication_list(medications, req.patient_data)
            change_recommendations = medication_advisor.recommend_medication_changes(medications, req.patient_data)
            optimized_regimen = medication_advisor.optimize_medication_regimen(req.patient_data)
            adherence_assessment = medication_advisor.check_medication_adherence(req.patient_data)
            medication_summary = medication_advisor.generate_medication_summary(req.patient_data)
            
            return {
                "status": "success",
                "medication_analysis": {
                    "current_analysis": medication_analysis,
                    "change_recommendations": change_recommendations,
                    "optimized_regimen": optimized_regimen,
                    "adherence_assessment": adherence_assessment,
                    "medication_summary": medication_summary
                },
                "timestamp": "2024-01-01T00:00:00Z"
            }
        else:
            # Basic analysis
            medication_analysis = medication_advisor.analyze_medication_list(medications, req.patient_data)
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
        vitals = {
            'blood_pressure': req.patient_data.get('vitals_bp', 0),
            'heart_rate': req.patient_data.get('vitals_hr', 0),
            'temperature': req.patient_data.get('temperature', 98.6),
            'oxygen_saturation': req.patient_data.get('oxygen_saturation', 98),
            'respiratory_rate': req.patient_data.get('respiratory_rate', 16)
        }
        
        if req.monitoring_type == "comprehensive":
            # Comprehensive monitoring
            vital_analysis = patient_monitor.analyze_vital_signs(vitals, req.patient_data)
            monitoring_plan = patient_monitor.generate_monitoring_plan(req.patient_data)
            medication_effects = patient_monitor.check_medication_effects(req.patient_data)
            patient_summary = patient_monitor.generate_patient_summary(req.patient_data)
            
            return {
                "status": "success",
                "patient_monitoring": {
                    "vital_analysis": vital_analysis,
                    "monitoring_plan": monitoring_plan,
                    "medication_effects": medication_effects,
                    "patient_summary": patient_summary
                },
                "timestamp": "2024-01-01T00:00:00Z"
            }
        else:
            # Basic monitoring
            vital_analysis = patient_monitor.analyze_vital_signs(vitals, req.patient_data)
            return {
                "status": "success",
                "vital_analysis": vital_analysis,
                "timestamp": "2024-01-01T00:00:00Z"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patient monitoring failed: {e}")

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
    return {
        "status": "healthy",
        "tools_available": 5,
        "tools": [
            "HealthAnalyzer",
            "ClinicalAdvisor", 
            "RiskAssessor",
            "MedicationAdvisor",
            "PatientMonitor"
        ]
    }