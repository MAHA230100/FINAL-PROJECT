"""Patient Feedback - Prompts"""

def build_feedback_analysis_prompt(data: dict) -> str:
    feedback = data.get('feedback_text', '')
    patient_name = data.get('patient_name', 'Patient')
    
    return f"""You are a patient experience AI assistant. Analyze the following feedback from {patient_name}.

PATIENT FEEDBACK:
{feedback}

OUTPUT FORMAT (strict JSON):
{{
  "sentiment": "<positive|neutral|negative>",
  "categories": ["<category 1>", "<category 2>"],
  "action_items": [
    {{"category": "<category>", "action": "<action>", "priority": "<low|medium|high>"}}
  ],
  "priority": "<low|medium|high>",
  "confidence": <float 0.0-1.0>,
  "summary": "<brief summary>"
}}

Classify sentiment, identify categories (e.g., wait time, staff, care quality), and suggest actions. Return ONLY valid JSON."""
