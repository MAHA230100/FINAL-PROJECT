"""
Patient Risk Assessment Feature - Main Processor
"""

from typing import Dict, Any, Optional, List
from ...core import BaseAIFeature, MissingDataResponse, validate_required_fields
from .schemas import PatientRiskOutputSchema
from .prefetch import prefetch_patient_risk_data
from .prompts import build_risk_assessment_prompt


class PatientRiskFeature(BaseAIFeature):
    """Patient Risk Assessment AI Feature"""
    
    def __init__(self):
        super().__init__("patient_risk")
    
    @property
    def required_fields(self) -> List[str]:
        return ["age", "gender", "vitals_bp", "vitals_hr"]
    
    @property
    def output_schema(self):
        return PatientRiskOutputSchema
    
    async def prefetch(self, patient_id: str, **kwargs) -> Dict[str, Any]:
        """Fetch patient data from database"""
        return await prefetch_patient_risk_data(patient_id)
    
    def validate_input(self, data: Dict[str, Any]) -> Optional[MissingDataResponse]:
        """Validate that required fields are present"""
        return validate_required_fields(data, self.required_fields, self.feature_name)
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        """Build AI prompt"""
        return build_risk_assessment_prompt(data)
    
    def post_process(self, ai_response: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process AI response and add metadata.
        Note: AI response is already validated against schema by AIService
        """
        # AI response is already parsed and validated
        # Just add any additional metadata
        return {
            **ai_response,
            "patient_id": data.get("patient_id"),
            "assessment_type": "comprehensive"
        }
