"""
Patient Risk Assessment Feature - Prompt Templates
"""


def build_risk_assessment_prompt(data: dict) -> str:
    """
    Build prompt for patient risk assessment.
    
    Args:
        data: Validated patient data
        
    Returns:
        Formatted prompt string
    """
    age = data.get('age', 'unknown')
    gender = data.get('gender', 'unknown')
    bp = data.get('vitals_bp', 'N/A')
    hr = data.get('vitals_hr', 'N/A')
    prev_admissions = data.get('previous_admissions', 0)
    comorbidities = data.get('comorbidities', [])
    lab_results = data.get('lab_results', 'Normal')
    
    # Format comorbidities
    if isinstance(comorbidities, list):
        comorb_str = ", ".join(comorbidities) if comorbidities else "None"
    else:
        comorb_str = str(comorbidities)
    
    prompt = f"""You are a clinical AI assistant performing a comprehensive patient risk assessment.

PATIENT PROFILE:
- Age: {age} years old
- Gender: {gender}
- Blood Pressure: {bp} mmHg
- Heart Rate: {hr} bpm
- Previous Hospital Admissions: {prev_admissions}
- Comorbidities: {comorb_str}
- Lab Results: {lab_results}

TASK:
Perform a detailed risk assessment analyzing:
1. Overall risk score (0-100)
2. Risk level classification (Low/Medium/High)
3. Key risk factors with severity
4. Specific recommendations for risk mitigation

OUTPUT FORMAT (strict JSON):
{{
  "risk_score": <integer 0-100>,
  "risk_level": "<Low|Medium|High>",
  "risk_factors": [
    {{
      "category": "<risk category>",
      "severity": "<low|medium|high>",
      "description": "<brief description>"
    }}
  ],
  "recommendations": [
    {{
      "priority": "<low|medium|high>",
      "action": "<specific action>",
      "rationale": "<why this matters>"
    }}
  ],
  "confidence": <float 0.0-1.0>,
  "summary": "<brief 2-3 sentence summary>"
}}

Provide clinical, evidence-based analysis. Return ONLY valid JSON."""
    
    return prompt
