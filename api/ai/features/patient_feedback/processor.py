"""Patient Feedback - Processor"""
from typing import Dict, Any, Optional, List
from ...core import BaseAIFeature, MissingDataResponse, validate_required_fields
from .schemas import PatientFeedbackOutputSchema
from .prefetch import prefetch_feedback_data
from .prompts import build_feedback_analysis_prompt

class PatientFeedbackFeature(BaseAIFeature):
    def __init__(self):
        super().__init__("patient_feedback")
    
    @property
    def required_fields(self) -> List[str]:
        return ["feedback_text"]
    
    @property
    def output_schema(self):
        return PatientFeedbackOutputSchema
    
    async def prefetch(self, patient_id: str, **kwargs) -> Dict[str, Any]:
        feedback_text = kwargs.get("feedback_text")
        return await prefetch_feedback_data(patient_id, feedback_text)
    
    def validate_input(self, data: Dict[str, Any]) -> Optional[MissingDataResponse]:
        validation = validate_required_fields(data, self.required_fields, self.feature_name)
        if validation:
            return validation
        
        if len(data.get("feedback_text", "")) < 5:
            return MissingDataResponse(
                feature=self.feature_name,
                missing_fields=["feedback_text"],
                user_message="Feedback text is too short. Please provide more details."
            )
        return None
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        return build_feedback_analysis_prompt(data)
    
    def post_process(self, ai_response: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**ai_response, "patient_id": data.get("patient_id")}
