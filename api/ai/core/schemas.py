"""
Core schemas for AI features.
Provides standard response formats and validation models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class AIStatus(str, Enum):
    """Standard status codes for AI responses"""
    SUCCESS = "success"
    ERROR = "error"
    MISSING_DATA = "missing_data"
    VALIDATION_ERROR = "validation_error"


class BaseAIResponse(BaseModel):
    """Standard response wrapper for all AI features"""
    status: AIStatus
    feature: str
    data: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ErrorResponse(BaseModel):
    """User-friendly error response"""
    status: Literal[AIStatus.ERROR] = AIStatus.ERROR
    feature: str
    message: str
    details: Optional[str] = None
    user_message: str  # Always user-friendly
    
    class Config:
        use_enum_values = True


class MissingDataResponse(BaseModel):
    """Response when required DB data is missing"""
    status: Literal[AIStatus.MISSING_DATA] = AIStatus.MISSING_DATA
    feature: str
    missing_fields: List[str]
    user_message: str = "Required patient data not available. Please provide details before proceeding."
    
    class Config:
        use_enum_values = True


class ValidationError(BaseModel):
    """Input validation error"""
    status: Literal[AIStatus.VALIDATION_ERROR] = AIStatus.VALIDATION_ERROR
    feature: str
    field: str
    message: str
    user_message: str
    
    class Config:
        use_enum_values = True
