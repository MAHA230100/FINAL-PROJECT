"""
Clinical Advisor - AI tool for providing clinical recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class ClinicalAdvisor:
    """AI-powered clinical advisor for healthcare recommendations"""
    
    def __init__(self):
        self.clinical_guidelines = {
            'hypertension': {
                'threshold': 140,
                'recommendations': [
                    'Monitor blood pressure every 4 hours',
                    'Consider antihypertensive medication',
                    'Dietary counseling for low-sodium diet',
                    'Regular exercise recommendations'
                ]
            },
            'diabetes': {
                'threshold': 126,
                'recommendations': [
                    'Blood glucose monitoring',
                    'HbA1c testing every 3 months',
                    'Dietary counseling',
                    'Foot care education'
                ]
            },
            'heart_disease': {
                'risk_factors': ['age', 'smoking', 'family_history'],
                'recommendations': [
                    'Cardiac monitoring',
                    'Echocardiogram if indicated',
                    'Lifestyle modifications',
                    'Medication review'
                ]
            }
        }
    
    def analyze_vital_signs(self, vitals: Dict[str, float]) -> Dict[str, Any]:
        """Analyze vital signs and provide recommendations"""
        analysis = {
            'vital_signs': vitals,
            'abnormalities': [],
            'recommendations': [],
            'priority_level': 'Normal'
        }
        
        # Blood pressure analysis
        bp = vitals.get('blood_pressure', 0)
        if bp > 140:
            analysis['abnormalities'].append('Hypertension')
            analysis['recommendations'].extend(self.clinical_guidelines['hypertension']['recommendations'])
            analysis['priority_level'] = 'High'
        elif bp > 120:
            analysis['abnormalities'].append('Elevated blood pressure')
            analysis['recommendations'].append('Monitor blood pressure closely')
            analysis['priority_level'] = 'Medium'
        
        # Heart rate analysis
        hr = vitals.get('heart_rate', 0)
        if hr > 100:
            analysis['abnormalities'].append('Tachycardia')
            analysis['recommendations'].append('Cardiac evaluation recommended')
            if analysis['priority_level'] == 'Normal':
                analysis['priority_level'] = 'Medium'
        elif hr < 60:
            analysis['abnormalities'].append('Bradycardia')
            analysis['recommendations'].append('Monitor heart rate closely')
            if analysis['priority_level'] == 'Normal':
                analysis['priority_level'] = 'Medium'
        
        # Temperature analysis
        temp = vitals.get('temperature', 0)
        if temp > 100.4:
            analysis['abnormalities'].append('Fever')
            analysis['recommendations'].append('Infection workup recommended')
            analysis['priority_level'] = 'High'
        elif temp < 95:
            analysis['abnormalities'].append('Hypothermia')
            analysis['recommendations'].append('Warming measures required')
            analysis['priority_level'] = 'High'
        
        # Oxygen saturation analysis
        spo2 = vitals.get('oxygen_saturation', 0)
        if spo2 < 90:
            analysis['abnormalities'].append('Hypoxemia')
            analysis['recommendations'].append('Oxygen therapy required')
            analysis['priority_level'] = 'High'
        elif spo2 < 95:
            analysis['abnormalities'].append('Mild hypoxemia')
            analysis['recommendations'].append('Monitor oxygen saturation')
            if analysis['priority_level'] == 'Normal':
                analysis['priority_level'] = 'Medium'
        
        return analysis
    
    def recommend_treatment_plan(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend treatment plan based on patient data"""
        treatment_plan = {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'diagnosis': patient_data.get('diagnosis', 'Unknown'),
            'treatment_recommendations': [],
            'medications': [],
            'monitoring_requirements': [],
            'follow_up_schedule': [],
            'discharge_criteria': []
        }
        
        # Analyze diagnosis and recommend treatments
        diagnosis = patient_data.get('diagnosis', '').lower()
        
        if 'pneumonia' in diagnosis:
            treatment_plan['treatment_recommendations'].extend([
                'Antibiotic therapy',
                'Respiratory support if needed',
                'Chest physiotherapy',
                'Fluid management'
            ])
            treatment_plan['medications'].append('Antibiotics (based on culture results)')
            treatment_plan['monitoring_requirements'].extend([
                'Vital signs every 4 hours',
                'Oxygen saturation monitoring',
                'Chest X-ray follow-up'
            ])
        
        elif 'heart_failure' in diagnosis:
            treatment_plan['treatment_recommendations'].extend([
                'Diuretic therapy',
                'ACE inhibitor or ARB',
                'Beta-blocker if tolerated',
                'Fluid restriction'
            ])
            treatment_plan['medications'].extend([
                'Furosemide',
                'Lisinopril or Losartan',
                'Metoprolol'
            ])
            treatment_plan['monitoring_requirements'].extend([
                'Daily weights',
                'Fluid intake/output',
                'Cardiac monitoring'
            ])
        
        elif 'diabetes' in diagnosis:
            treatment_plan['treatment_recommendations'].extend([
                'Blood glucose monitoring',
                'Insulin therapy if needed',
                'Dietary counseling',
                'Foot care education'
            ])
            treatment_plan['medications'].append('Insulin or oral hypoglycemics')
            treatment_plan['monitoring_requirements'].extend([
                'Blood glucose 4 times daily',
                'HbA1c every 3 months',
                'Foot examination'
            ])
        
        # General recommendations based on age and comorbidities
        age = patient_data.get('age', 0)
        if age > 65:
            treatment_plan['treatment_recommendations'].append('Geriatric assessment')
            treatment_plan['monitoring_requirements'].append('Fall risk assessment')
        
        # Comorbidities consideration
        comorbidities = patient_data.get('comorbidities', '')
        if comorbidities:
            comorbidity_list = comorbidities.split(',')
            for comorbidity in comorbidity_list:
                comorbidity = comorbidity.strip().lower()
                if 'hypertension' in comorbidity:
                    treatment_plan['medications'].append('Antihypertensive medication')
                elif 'diabetes' in comorbidity:
                    treatment_plan['monitoring_requirements'].append('Blood glucose monitoring')
        
        # Set follow-up schedule
        if treatment_plan['priority_level'] == 'High':
            treatment_plan['follow_up_schedule'].append('Daily physician rounds')
        else:
            treatment_plan['follow_up_schedule'].append('Every 2-3 days physician rounds')
        
        # Set discharge criteria
        treatment_plan['discharge_criteria'] = [
            'Stable vital signs for 24 hours',
            'Patient able to perform activities of daily living',
            'Medication compliance demonstrated',
            'Follow-up appointments scheduled'
        ]
        
        return treatment_plan
    
    def assess_drug_interactions(self, medications: List[str]) -> Dict[str, Any]:
        """Assess potential drug interactions"""
        interactions = {
            'medications': medications,
            'interactions': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Common drug interactions
        interaction_pairs = {
            ('warfarin', 'aspirin'): 'Increased bleeding risk',
            ('warfarin', 'ibuprofen'): 'Increased bleeding risk',
            ('digoxin', 'furosemide'): 'Digoxin toxicity risk',
            ('ace_inhibitor', 'potassium'): 'Hyperkalemia risk',
            ('metformin', 'contrast_dye'): 'Lactic acidosis risk'
        }
        
        for i, med1 in enumerate(medications):
            for j, med2 in enumerate(medications[i+1:], i+1):
                med1_lower = med1.lower()
                med2_lower = med2.lower()
                
                for (drug1, drug2), warning in interaction_pairs.items():
                    if (drug1 in med1_lower and drug2 in med2_lower) or \
                       (drug2 in med1_lower and drug1 in med2_lower):
                        interactions['interactions'].append({
                            'medication1': med1,
                            'medication2': med2,
                            'interaction': warning,
                            'severity': 'High' if 'bleeding' in warning or 'toxicity' in warning else 'Medium'
                        })
        
        # Generate recommendations
        if interactions['interactions']:
            interactions['recommendations'].append('Review medication list with pharmacist')
            interactions['recommendations'].append('Monitor for adverse effects')
        else:
            interactions['recommendations'].append('No significant drug interactions identified')
        
        return interactions
    
    def generate_discharge_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive discharge summary"""
        discharge_summary = {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'admission_date': patient_data.get('admission_date', 'Unknown'),
            'discharge_date': datetime.now().strftime('%Y-%m-%d'),
            'length_of_stay': patient_data.get('length_of_stay', 0),
            'primary_diagnosis': patient_data.get('diagnosis', 'Unknown'),
            'secondary_diagnoses': patient_data.get('comorbidities', 'None'),
            'treatment_summary': [],
            'discharge_medications': [],
            'follow_up_instructions': [],
            'patient_education': [],
            'prognosis': 'Good'
        }
        
        # Treatment summary based on diagnosis
        diagnosis = patient_data.get('diagnosis', '').lower()
        if 'pneumonia' in diagnosis:
            discharge_summary['treatment_summary'].append('Antibiotic therapy completed')
            discharge_summary['follow_up_instructions'].append('Follow-up chest X-ray in 2 weeks')
        elif 'heart_failure' in diagnosis:
            discharge_summary['treatment_summary'].append('Heart failure management optimized')
            discharge_summary['follow_up_instructions'].append('Cardiology follow-up in 1 week')
        
        # Discharge medications
        discharge_summary['discharge_medications'] = [
            'Continue home medications as prescribed',
            'New medications as discussed with patient'
        ]
        
        # Patient education
        discharge_summary['patient_education'] = [
            'Medication compliance instructions',
            'Warning signs to watch for',
            'When to seek medical attention',
            'Lifestyle modifications as discussed'
        ]
        
        # Prognosis based on outcome
        outcome = patient_data.get('outcome', '')
        if outcome == 'Discharged':
            discharge_summary['prognosis'] = 'Good'
        elif outcome == 'Readmitted':
            discharge_summary['prognosis'] = 'Guarded - requires close monitoring'
        
        return discharge_summary
    
    def provide_clinical_guidance(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide comprehensive clinical guidance"""
        guidance = {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'clinical_assessment': {},
            'treatment_recommendations': [],
            'monitoring_plan': [],
            'risk_factors': [],
            'preventive_measures': [],
            'follow_up_plan': []
        }
        
        # Clinical assessment
        age = patient_data.get('age', 0)
        vitals = {
            'blood_pressure': patient_data.get('vitals_bp', 0),
            'heart_rate': patient_data.get('vitals_hr', 0),
            'temperature': patient_data.get('temperature', 98.6),
            'oxygen_saturation': patient_data.get('oxygen_saturation', 98)
        }
        
        vital_analysis = self.analyze_vital_signs(vitals)
        guidance['clinical_assessment'] = vital_analysis
        
        # Treatment recommendations
        treatment_plan = self.recommend_treatment_plan(patient_data)
        guidance['treatment_recommendations'] = treatment_plan['treatment_recommendations']
        
        # Risk factors
        if age > 65:
            guidance['risk_factors'].append('Advanced age')
        if vitals['blood_pressure'] > 140:
            guidance['risk_factors'].append('Hypertension')
        if patient_data.get('comorbidities'):
            guidance['risk_factors'].append('Multiple comorbidities')
        
        # Preventive measures
        guidance['preventive_measures'] = [
            'Regular health screenings',
            'Vaccination updates',
            'Lifestyle modifications',
            'Medication adherence'
        ]
        
        # Follow-up plan
        guidance['follow_up_plan'] = [
            'Primary care follow-up in 1-2 weeks',
            'Specialist consultation if needed',
            'Laboratory follow-up as indicated',
            'Patient education reinforcement'
        ]
        
        return guidance
