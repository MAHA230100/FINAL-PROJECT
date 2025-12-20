"""
Unified AI API Router
Provides centralized endpoints for all AI features.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from .core import ai_service, BaseAIResponse, ErrorResponse, AIStatus
from .features.patient_risk import PatientRiskFeature
from .features.notes_summarizer import NotesSummarizerFeature
from .features.image_diagnostics import ImageDiagnosticsFeature
from .features.patient_feedback import PatientFeedbackFeature
from .registry import registry

# Register all features
registry.register("patient_risk", PatientRiskFeature)
registry.register("notes_summarizer", NotesSummarizerFeature)
registry.register("image_diagnostics", ImageDiagnosticsFeature)
registry.register("patient_feedback", PatientFeedbackFeature)

router = APIRouter(prefix="/ai-v2", tags=["AI Features V2"])


# Request models
class PatientRiskRequest(BaseModel):
    patient_id: str


class NotesSummarizerRequest(BaseModel):
    patient_id: str


class ImageDiagnosticsRequest(BaseModel):
    patient_id: str
    image_type: str = "x-ray"


class PatientFeedbackRequest(BaseModel):
    patient_id: str
    feedback_text: str


async def process_feature(feature_name: str, patient_id: str, **kwargs) -> Dict[str, Any]:
    """
    Generic feature processing flow: prefetch → validate → prompt → AI → process
    
    Args:
        feature_name: Name of registered feature
        patient_id: Patient identifier
        **kwargs: Additional feature-specific args
        
    Returns:
        Standardized response dict
    """
    try:
        # Get feature class
        feature_class = registry.get(feature_name)
        feature = feature_class()
        
        # Step 1: Prefetch data
        data = await feature.prefetch(patient_id, **kwargs)
        
        # Step 2: Validate input
        validation_error = feature.validate_input(data)
        if validation_error:
            return validation_error.dict()
        
        # Step 3: Build prompt
        prompt = feature.build_prompt(data)
        
        # Step 4: Call AI
        ai_result = await ai_service.generate(
            prompt=prompt,
            feature=feature_name,
            schema_class=feature.output_schema
        )
        
        # Step 5: Check for errors
        if ai_result["status"] != AIStatus.SUCCESS:
            return ai_result["error"]
        
        # Step 6: Post-process
        processed_data = feature.post_process(ai_result["data"], data)
        
        # Step 7: Return standardized response
        return BaseAIResponse(
            status=AIStatus.SUCCESS,
            feature=feature_name,
            data=processed_data,
            confidence=processed_data.get("confidence")
        ).dict()
        
    except ValueError as e:
        # Unknown feature
        return ErrorResponse(
            feature=feature_name,
            message="UNKNOWN_FEATURE",
            details=str(e),
            user_message="This feature is not available."
        ).dict()
    except Exception as e:
        # Unexpected error
        return ErrorResponse(
            feature=feature_name,
            message="INTERNAL_ERROR",
            details=str(e),
            user_message="An unexpected error occurred. Please try again."
        ).dict()


@router.post("/patient-risk")
async def assess_patient_risk(req: PatientRiskRequest):
    """Assess patient health risks"""
    return await process_feature("patient_risk", req.patient_id)


@router.post("/summarize-notes")
async def summarize_notes(req: NotesSummarizerRequest):
    """Summarize patient clinical notes"""
    return await process_feature("notes_summarizer", req.patient_id)


@router.post("/analyze-image")
async def analyze_image(req: ImageDiagnosticsRequest):
    """Analyze medical imaging"""
    return await process_feature("image_diagnostics", req.patient_id, image_type=req.image_type)


@router.post("/analyze-feedback")
async def analyze_feedback(req: PatientFeedbackRequest):
    """Analyze patient feedback"""
    return await process_feature("patient_feedback", req.patient_id, feedback_text=req.feedback_text)


@router.get("/features")
async def list_features():
    """List all available AI features"""
    return {
        "features": registry.list_features(),
        "count": len(registry.list_features())
    }
