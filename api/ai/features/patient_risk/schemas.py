"""
Patient Risk Assessment Feature - Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class RiskFactorSchema(BaseModel):
    """Individual risk factor"""
    category: str
    severity: Literal["low", "medium", "high"]
    description: str


class RiskRecommendationSchema(BaseModel):
    """Risk mitigation recommendation"""
    priority: Literal["low", "medium", "high"]
    action: str
    rationale: str


class PatientRiskOutputSchema(BaseModel):
    """Output schema for patient risk assessment"""
    risk_score: int = Field(..., ge=0, le=100, description="Overall risk score 0-100")
    risk_level: Literal["Low", "Medium", "High"]
    risk_factors: List[RiskFactorSchema]
    recommendations: List[RiskRecommendationSchema]
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=10, description="Brief risk summary")
