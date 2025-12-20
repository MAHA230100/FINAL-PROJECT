"""Image Diagnostics - Prompts"""

def build_image_diagnostics_prompt(data: dict) -> str:
    patient_name = data.get('patient_name', 'Patient')
    age = data.get('age', 'unknown')
    gender = data.get('gender', 'unknown')
    image_type = data.get('image_type', 'Medical Image')
    
    return f"""You are a radiologist AI assistant analyzing medical imaging.

PATIENT CONTEXT:
- Name: {patient_name}
- Age: {age}, Gender: {gender}
- Image Type: {image_type}

TASK: Generate a simulated radiology report for this {image_type}.

OUTPUT FORMAT (strict JSON):
{{
  "findings": [
    {{"location": "<anatomical location>", "description": "<finding>", "severity": "<minor|moderate|significant>"}}
  ],
  "impression": "<clinical impression>",
  "recommendations": ["<recommendation 1>", "<recommendation 2>"],
  "image_quality": "<good|acceptable|poor>",
  "confidence": <float 0.0-1.0>
}}

Provide realistic clinical findings. Return ONLY valid JSON."""
