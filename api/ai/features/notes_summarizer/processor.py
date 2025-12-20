"""Notes Summarizer - Processor"""
from typing import Dict, Any, Optional, List
from ...core import BaseAIFeature, MissingDataResponse, validate_required_fields
from .schemas import NotesSummarizerOutputSchema
from .prefetch import prefetch_notes_data
from .prompts import build_notes_summary_prompt

class NotesSummarizerFeature(BaseAIFeature):
    def __init__(self):
        super().__init__("notes_summarizer")
    
    @property
    def required_fields(self) -> List[str]:
        return ["notes_text"]
    
    @property
    def output_schema(self):
        return NotesSummarizerOutputSchema
    
    async def prefetch(self, patient_id: str, **kwargs) -> Dict[str, Any]:
        return await prefetch_notes_data(patient_id)
    
    def validate_input(self, data: Dict[str, Any]) -> Optional[MissingDataResponse]:
        validation = validate_required_fields(data, self.required_fields, self.feature_name)
        if validation:
            return validation
        
        # Additional validation: notes must have minimum length
        if len(data.get("notes_text", "")) < 10:
            return MissingDataResponse(
                feature=self.feature_name,
                missing_fields=["notes_text"],
                user_message="No clinical notes found for this patient. Please add notes before summarizing."
            )
        return None
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        return build_notes_summary_prompt(data)
    
    def post_process(self, ai_response: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**ai_response, "patient_id": data.get("patient_id")}
