"""Image Diagnostics - Processor"""
from typing import Dict, Any, Optional, List
from ...core import BaseAIFeature, MissingDataResponse, validate_required_fields
from .schemas import ImageDiagnosticsOutputSchema
from .prefetch import prefetch_image_data
from .prompts import build_image_diagnostics_prompt

class ImageDiagnosticsFeature(BaseAIFeature):
    ALLOWED_IMAGE_TYPES = ["x-ray", "ct", "mri", "ultrasound", "xray"]
    
    def __init__(self):
        super().__init__("image_diagnostics")
    
    @property
    def required_fields(self) -> List[str]:
        return ["patient_name", "image_type"]
    
    @property
    def output_schema(self):
        return ImageDiagnosticsOutputSchema
    
    async def prefetch(self, patient_id: str, **kwargs) -> Dict[str, Any]:
        image_type = kwargs.get("image_type")
        return await prefetch_image_data(patient_id, image_type)
    
    def validate_input(self, data: Dict[str, Any]) -> Optional[MissingDataResponse]:
        validation = validate_required_fields(data, self.required_fields, self.feature_name)
        if validation:
            return validation
        
        # Validate image type
        image_type = str(data.get("image_type", "")).lower()
        if image_type not in self.ALLOWED_IMAGE_TYPES:
            return MissingDataResponse(
                feature=self.feature_name,
                missing_fields=["image_type"],
                user_message=f"Invalid medical image type. Accepted: {', '.join(self.ALLOWED_IMAGE_TYPES)}"
            )
        return None
    
    def build_prompt(self, data: Dict[str, Any]) -> str:
        return build_image_diagnostics_prompt(data)
    
    def post_process(self, ai_response: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**ai_response, "patient_id": data.get("patient_id"), "image_type": data.get("image_type")}
