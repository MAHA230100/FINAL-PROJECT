"""Notes Summarizer - Prompts"""

def build_notes_summary_prompt(data: dict) -> str:
    notes = data.get('notes_text', '')
    patient_name = data.get('patient_name', 'Patient')
    
    return f"""You are a clinical AI assistant. Summarize the following clinical notes for {patient_name}.

CLINICAL NOTES:
{notes}

OUTPUT FORMAT (strict JSON):
{{
  "summary": "<concise 2-3 paragraph summary>",
  "key_points": [
    {{"category": "<category>", "point": "<key point>"}}
  ],
  "sentiment": "<positive|neutral|negative>",
  "confidence": <float 0.0-1.0>
}}

Extract key findings, diagnoses, and treatment plans. Return ONLY valid JSON."""
