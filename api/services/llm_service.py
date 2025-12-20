import os
import json
from typing import Optional, Dict, Any, List
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.model = None
        
        # Debug: Print what we're getting
        print(f"🔍 Debug: GOOGLE_API_KEY value: {self.api_key[:10] + '...' if self.api_key else 'None'}")
        print(f"🔍 Debug: genai module: {'Available' if genai else 'Not available'}")
        
        if self.api_key and genai:
            try:
                genai.configure(api_key=self.api_key)
                
                # Try multiple model names in order of preference
                # gemini-3-flash-preview: Confirmed working via Postman test
                model_names = [
                    'gemini-3-flash-preview',
                    'gemini-flash-latest',
                    'gemini-2.5-flash',
                    'gemini-2.5-pro'
                ]
                
                for model_name in model_names:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        print(f"✅ Gemini LLM Initialized ({model_name})")
                        break
                    except Exception as model_err:
                        print(f"⚠️ Failed to load {model_name}: {str(model_err)[:100]}")
                        continue
                
                if not self.model:
                    print("❌ All model attempts failed. Using Mock Mode.")
                    
            except Exception as e:
                print(f"⚠️ Gemini Setup Failed: {e}")
        else:
            if not self.api_key:
                print("ℹ️ Google API Key not found. Using Mock Mode.")
            elif not genai:
                print("ℹ️ google-generativeai library not installed. Using Mock Mode.")

    def is_active(self) -> bool:
        return self.model is not None

    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate a text response given a prompt and optional context.
        """
        if not self.is_active():
            return self._mock_response(prompt)
            
        try:
            full_prompt = f"{context}\n\nQuery: {prompt}"
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

    def analyze_patient(self, patient_data: Dict[str, Any], task: str) -> str:
        """
        Analyze specific patient JSON data for a given task.
        """
        if not self.is_active():
            return self._mock_response(task, patient_name=patient_data.get('name'))

        # Sanitize/Format Patient Data
        p_text = json.dumps(patient_data, indent=2)
        
        system_instruction = (
            "You are an expert medical AI assistant. You are analyzing the following patient data:\n"
            f"```json\n{p_text}\n```\n"
            "Provide a professional, clinical analysis."
        )
        
        try:
            response = self.model.generate_content(f"{system_instruction}\n\nTask: {task}")
            return response.text
        except Exception as e:
            return f"Error analyzing patient: {e}"

    def _mock_response(self, query: str, patient_name: str = "the patient") -> str:
        """Fallback mock responses when no API key is present."""
        if "summarize" in query.lower():
            return f"[MOCK] Summary for {patient_name}: Patient condition appears stable based on available notes. Suggested monitoring for vitals."
        elif "risk" in query.lower():
            return f"[MOCK] Risk Assessment for {patient_name}: Moderate risk of readmission due to age and history. Recommend follow-up in 7 days."
        elif "feedback" in query.lower():
            return f"[MOCK] Feedback Analysis: Sentiment is positive. Key detail: '{query[:20]}...'."
        else:
            return f"[MOCK] Processing query for {patient_name}: '{query}'. (Simulated response - Add GOOGLE_API_KEY to enable Real AI)"

# Singleton Instance
llm_service = LLMService()
