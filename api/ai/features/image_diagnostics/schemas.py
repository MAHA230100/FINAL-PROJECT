"""Image Diagnostics - Schemas"""
from pydantic import BaseModel, Field
from typing import List, Literal

class FindingSchema(BaseModel):
    location: str
    description: str
    severity: Literal["minor", "moderate", "significant"]

class ImageDiagnosticsOutputSchema(BaseModel):
    findings: List[FindingSchema]
    impression: str = Field(..., min_length=20)
    recommendations: List[str]
    image_quality: Literal["good", "acceptable", "poor"]
    confidence: float = Field(..., ge=0.0, le=1.0)
