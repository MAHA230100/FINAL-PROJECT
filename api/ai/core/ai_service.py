"""
Centralized AI Service.
Single entry point for all AI calls with standardized error handling.
"""

import os
from typing import Optional, Dict, Any
import json

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .schemas import BaseAIResponse, ErrorResponse, AIStatus
from .validators import clean_ai_response


class AIService:
    """Central AI service managing all LLM interactions"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = None
        self._initialized = False
        
        if self.api_key and genai:
            try:
                genai.configure(api_key=self.api_key)
                
                # Try models in order of preference
                model_names = [
                    'gemini-3-flash-preview',
                    'gemini-flash-latest',
                    'gemini-2.5-flash',
                ]
                
                for model_name in model_names:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        print(f"✅ AI Service Initialized ({model_name})")
                        self._initialized = True
                        break
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"⚠️ AI Service Init Failed: {e}")
    
    def is_active(self) -> bool:
        """Check if AI service is ready"""
        return self._initialized and self.model is not None
    
    async def generate(
        self, 
        prompt: str, 
        feature: str,
        schema_class: Optional[type] = None,
        temperature: float = 1.0,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate content with error handling and retry logic.
        
        Args:
            prompt: Prompt text
            feature: Feature name for error reporting
            schema_class: Optional Pydantic schema for validation
            temperature: Generation temperature
            max_retries: Number of retry attempts
            
        Returns:
            Dictionary with status and data/error
        """
        if not self.is_active():
            return self._create_error(
                feature,
                "AI_ERROR",
                "AI service not initialized"
            )
        
        for attempt in range(max_retries):
            try:
                # Generate content
                response = self.model.generate_content(prompt)
                text = response.text
                
                # Clean response
                text = clean_ai_response(text)
                
                # Try to parse as JSON if schema provided
                if schema_class:
                    try:
                        data = json.loads(text)
                        # Validate against schema
                        validated = schema_class(**data)
                        return {
                            "status": AIStatus.SUCCESS,
                            "data": validated.dict()
                        }
                    except json.JSONDecodeError:
                        if attempt < max_retries - 1:
                            continue  # Retry
                        return self._create_error(
                            feature,
                            "SCHEMA_ERROR",
                            f"Invalid JSON response: {text[:100]}"
                        )
                    except Exception as e:
                        if attempt < max_retries - 1:
                            continue  # Retry
                        return self._create_error(
                            feature,
                            "SCHEMA_ERROR",
                            f"Schema validation failed: {str(e)}"
                        )
                else:
                    # Return raw text
                    return {
                        "status": AIStatus.SUCCESS,
                        "data": {"text": text}
                    }
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    continue  # Retry
                return self._create_error(
                    feature,
                    "AI_ERROR",
                    f"Generation failed: {str(e)}"
                )
        
        # Should not reach here
        return self._create_error(feature, "AI_ERROR", "Max retries exceeded")
    
    def _create_error(self, feature: str, error_type: str, details: str) -> Dict[str, Any]:
        """Create error response dict"""
        user_messages = {
            "AI_ERROR": "Analysis temporarily unavailable. Please try again later.",
            "SCHEMA_ERROR": "Unexpected response format. Please contact support.",
            "TIMEOUT": "Request took too long. Please try again.",
        }
        
        return {
            "status": AIStatus.ERROR,
            "error": {
                "feature": feature,
                "message": error_type,
                "details": details,
                "user_message": user_messages.get(error_type, "An error occurred. Please try again.")
            }
        }


# Singleton instance
ai_service = AIService()
