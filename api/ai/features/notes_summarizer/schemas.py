"""Notes Summarizer - Schemas"""
from pydantic import BaseModel, Field
from typing import List, Literal

class KeyPointSchema(BaseModel):
    category: str
    point: str

class NotesSummarizerOutputSchema(BaseModel):
    summary: str = Field(..., min_length=20)
    key_points: List[KeyPointSchema]
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float = Field(..., ge=0.0, le=1.0)
