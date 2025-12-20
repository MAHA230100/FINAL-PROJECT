"""Patient Feedback - Schemas"""
from pydantic import BaseModel, Field
from typing import List, Literal

class ActionItemSchema(BaseModel):
    category: str
    action: str
    priority: Literal["low", "medium", "high"]

class PatientFeedbackOutputSchema(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    categories: List[str]
    action_items: List[ActionItemSchema]
    priority: Literal["low", "medium", "high"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=10)
