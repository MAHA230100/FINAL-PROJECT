"""
Centralized AI Service.
Single entry point for all AI calls with standardized error handling.
"""

import os
from typing import Optional, Dict, Any
import json
import time

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .schemas import BaseAIResponse, ErrorResponse, AIStatus
from .validators import clean_ai_response


class AIService:
    """Central AI service managing all LLM interactions with automatic fallback"""
    
    # Model priority list - will try in order
    MODEL_PRIORITY = [
        'gemini-2.5-flash-lite',  # Primary - good quota
        'gemini-flash-latest',
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-3-flash-preview',
    ]
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = None
        self.current_model_name = None
        self._initialized = False
        self._failed_models = set()  # Track models that hit quota limits
        
        if self.api_key and genai:
            try:
                genai.configure(api_key=self.api_key)
                self._try_initialize_model()
            except Exception as e:
                print(f"⚠️ AI Service Init Failed: {e}")
    
    def _try_initialize_model(self):
        """Try to initialize with the best available model"""
        for model_name in self.MODEL_PRIORITY:
            if model_name in self._failed_models:
                continue
                
            try:
                self.model = genai.GenerativeModel(model_name)
                self.current_model_name = model_name
                print(f"✅ AI Service Initialized ({model_name})")
                self._initialized = True
                return True
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {str(e)[:100]}")
                continue
        
        print("❌ All models failed to initialize")
        return False
    
    def _switch_to_next_model(self):
        """Switch to the next available model after quota error"""
        print(f"⚠️ Switching away from {self.current_model_name} due to quota limit")
        self._failed_models.add(self.current_model_name)
        
        # Try next available model
        for model_name in self.MODEL_PRIORITY:
            if model_name not in self._failed_models:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    self.current_model_name = model_name
                    print(f"✅ Switched to model: {model_name}")
                    return True
                except Exception as e:
                    print(f"⚠️ Failed to switch to {model_name}: {str(e)[:100]}")
                    self._failed_models.add(model_name)
                    continue
        
        print("❌ No alternative models available")
        return False
    
    def is_active(self) -> bool:
        """Check if AI service is ready"""
        return self._initialized and self.model is not None
    
    async def generate(
        self, 
        prompt: str, 
        feature: str,
        schema_class: Optional[type] = None,
        temperature: float = 1.0,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate content with error handling, retry logic, and automatic model fallback.
        
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
                error_str = str(e)
                
                # Check for 429 quota error
                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    print(f"⚠️ Quota limit hit on {self.current_model_name}")
                    
                    # Try to switch to another model
                    if self._switch_to_next_model():
                        print(f"🔄 Retrying with {self.current_model_name}...")
                        continue  # Retry with new model
                    else:
                        return self._create_error(
                            feature,
                            "QUOTA_ERROR",
                            "All models exceeded quota limits. Please try again later."
                        )
                
                # Other errors - retry if attempts remain
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief delay before retry
                    continue
                    
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
            "QUOTA_ERROR": "API quota exceeded. Trying alternative models or please try again in a few minutes.",
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
