"""
Base feature class that all AI features must extend.
Enforces the standard contract: prefetch → validate → prompt → process
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from .schemas import BaseAIResponse, ErrorResponse, MissingDataResponse
from .validators import validate_required_fields


class BaseAIFeature(ABC):
    """
    Abstract base class for AI features.
    
    All features must implement:
    - prefetch(): Fetch required DB data
    - validate_input(): Validate inputs
    - build_prompt(): Construct AI prompt
    - output_schema: Pydantic schema for output
    - post_process(): Process AI response
    """
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
    
    @property
    @abstractmethod
    def required_fields(self) -> List[str]:
        """List of required fields that must be prefetched"""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Pydantic schema for output validation"""
        pass
    
    @abstractmethod
    async def prefetch(self, patient_id: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch required data from database.
        
        Args:
            patient_id: Patient identifier
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with prefetched data
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> Optional[MissingDataResponse]:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            MissingDataResponse if validation fails, None otherwise
        """
        pass
    
    @abstractmethod
    def build_prompt(self, data: Dict[str, Any]) -> str:
        """
        Build AI prompt from validated data.
        
        Args:
            data: Validated input data
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def post_process(self, ai_response: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process AI response and format output.
        
        Args:
            ai_response: Raw AI response
            data: Original input data
            
        Returns:
            Processed output dictionary
        """
        pass
    
    def create_error_response(self, message: str, details: str = None) -> ErrorResponse:
        """
        Create a user-friendly error response.
        
        Args:
            message: Technical error message
            details: Additional details
            
        Returns:
            ErrorResponse object
        """
        # Map technical errors to user-friendly messages
        user_messages = {
            "AI_ERROR": "Analysis temporarily unavailable. Please try again later.",
            "SCHEMA_ERROR": "Unexpected response format. Please contact support.",
            "TIMEOUT": "Request took too long. Please try again.",
        }
        
        user_message = user_messages.get(message, "An error occurred. Please try again.")
        
        return ErrorResponse(
            feature=self.feature_name,
            message=message,
            details=details,
            user_message=user_message
        )
