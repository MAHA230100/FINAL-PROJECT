"""
Validation utilities for AI features.
"""

from typing import List, Dict, Any, Optional
from pydantic import ValidationError as PydanticValidationError
from .schemas import ValidationError, MissingDataResponse


def validate_required_fields(data: Dict[str, Any], required_fields: List[str], feature: str) -> Optional[MissingDataResponse]:
    """
    Check if all required fields are present and non-empty.
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
        feature: Feature name for error reporting
        
    Returns:
        MissingDataResponse if fields are missing, None otherwise
    """
    missing = []
    
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            missing.append(field)
    
    if missing:
        return MissingDataResponse(
            feature=feature,
            missing_fields=missing,
            user_message=f"Required patient data not available: {', '.join(missing)}. Please provide details before proceeding."
        )
    
    return None


def validate_schema(data: Dict[str, Any], schema_class, feature: str) -> tuple[Optional[Any], Optional[ValidationError]]:
    """
    Validate data against a Pydantic schema.
    
    Args:
        data: Data to validate
        schema_class: Pydantic model class
        feature: Feature name for error reporting
        
    Returns:
        Tuple of (validated_data, error). One will be None.
    """
    try:
        validated = schema_class(**data)
        return validated, None
    except PydanticValidationError as e:
        # Get first error for user-friendly message
        first_error = e.errors()[0]
        field = ".".join(str(loc) for loc in first_error['loc'])
        
        return None, ValidationError(
            feature=feature,
            field=field,
            message=first_error['msg'],
            user_message=f"Invalid {field}: {first_error['msg']}"
        )


def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """Safely get nested dictionary values"""
    try:
        return data.get(key, default)
    except (AttributeError, KeyError):
        return default


def clean_ai_response(response: str) -> str:
    """
    Clean AI response by removing markdown code blocks.
    
    Args:
        response: Raw AI response string
        
    Returns:
        Cleaned response string
    """
    # Remove markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    
    return response.strip()
