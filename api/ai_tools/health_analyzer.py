"""
Health Data Analyzer - AI tool for analyzing healthcare data patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
from datetime import datetime, timedelta

class HealthAnalyzer:
    """AI-powered health data analyzer"""
    
    def __init__(self):
        self.risk_factors = {
            'age': {'high': 70, 'medium': 50},
            'blood_pressure': {'high': 140, 'medium': 120},
            'heart_rate': {'high': 100, 'medium': 80},
            'length_of_stay': {'high': 14, 'medium': 7}
        }
    
    def analyze_patient_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patient risk factors"""
        risk_score = 0
        risk_factors = []
        recommendations = []
        
        # Age analysis
        age = patient_data.get('age', 0)
        if age > self.risk_factors['age']['high']:
            risk_score += 3
            risk_factors.append('High age risk')
            recommendations.append('Consider geriatric care protocols')
        elif age > self.risk_factors['age']['medium']:
            risk_score += 1
            risk_factors.append('Medium age risk')
        
        # Blood pressure analysis
        bp = patient_data.get('vitals_bp', 0)
        if bp > self.risk_factors['blood_pressure']['high']:
            risk_score += 3
            risk_factors.append('Hypertension risk')
            recommendations.append('Monitor blood pressure closely')
        elif bp > self.risk_factors['blood_pressure']['medium']:
            risk_score += 1
            risk_factors.append('Elevated blood pressure')
        
        # Heart rate analysis
        hr = patient_data.get('vitals_hr', 0)
        if hr > self.risk_factors['heart_rate']['high']:
            risk_score += 2
            risk_factors.append('Tachycardia risk')
            recommendations.append('Cardiac monitoring recommended')
        elif hr < 60:
            risk_score += 1
            risk_factors.append('Bradycardia risk')
        
        # Length of stay analysis
        los = patient_data.get('length_of_stay', 0)
        if los > self.risk_factors['length_of_stay']['high']:
            risk_score += 2
            risk_factors.append('Extended stay risk')
            recommendations.append('Review discharge planning')
        
        # Comorbidities analysis
        comorbidities = patient_data.get('comorbidities', '')
        if comorbidities:
            comorbidity_count = len(comorbidities.split(','))
            risk_score += comorbidity_count
            risk_factors.append(f'{comorbidity_count} comorbidities')
        
        # Determine risk level
        if risk_score >= 7:
            risk_level = 'High'
        elif risk_score >= 4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def predict_readmission_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict readmission risk"""
        # Simple heuristic-based prediction
        risk_factors = 0
        
        # Age factor
        age = patient_data.get('age', 0)
        if age > 65:
            risk_factors += 1
        
        # Previous admissions
        prev_admissions = patient_data.get('previous_admissions', 0)
        if prev_admissions > 2:
            risk_factors += 2
        elif prev_admissions > 0:
            risk_factors += 1
        
        # Length of stay
        los = patient_data.get('length_of_stay', 0)
        if los > 10:
            risk_factors += 1
        
        # Comorbidities
        comorbidities = patient_data.get('comorbidities', '')
        if comorbidities:
            comorbidity_count = len(comorbidities.split(','))
            if comorbidity_count > 2:
                risk_factors += 2
            elif comorbidity_count > 0:
                risk_factors += 1
        
        # Calculate probability
        readmission_prob = min(0.9, risk_factors * 0.15)
        
        return {
            'readmission_probability': readmission_prob,
            'risk_level': 'High' if readmission_prob > 0.6 else 'Medium' if readmission_prob > 0.3 else 'Low',
            'risk_factors_count': risk_factors,
            'recommendations': self._get_readmission_recommendations(readmission_prob)
        }
    
    def _get_readmission_recommendations(self, probability: float) -> List[str]:
        """Get recommendations based on readmission probability"""
        if probability > 0.6:
            return [
                'Schedule follow-up appointment within 1 week',
                'Provide detailed discharge instructions',
                'Consider home health services',
                'Monitor patient closely post-discharge'
            ]
        elif probability > 0.3:
            return [
                'Schedule follow-up appointment within 2 weeks',
                'Provide discharge instructions',
                'Consider patient education materials'
            ]
        else:
            return [
                'Standard discharge procedures',
                'Routine follow-up as needed'
            ]
    
    def analyze_treatment_patterns(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze treatment patterns and effectiveness"""
        patterns = {
            'treatment_duration': patient_data.get('length_of_stay', 0),
            'treatment_effectiveness': 'Unknown',
            'care_quality_indicators': []
        }
        
        # Analyze treatment duration
        los = patient_data.get('length_of_stay', 0)
        if los < 3:
            patterns['care_quality_indicators'].append('Efficient treatment')
        elif los > 14:
            patterns['care_quality_indicators'].append('Extended treatment required')
        
        # Analyze outcome
        outcome = patient_data.get('outcome', '')
        if outcome == 'Discharged':
            patterns['treatment_effectiveness'] = 'Successful'
            patterns['care_quality_indicators'].append('Positive outcome')
        elif outcome == 'Readmitted':
            patterns['treatment_effectiveness'] = 'Requires monitoring'
            patterns['care_quality_indicators'].append('Readmission risk')
        
        return patterns
    
    def generate_insights(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive health insights"""
        risk_analysis = self.analyze_patient_risk(patient_data)
        readmission_risk = self.predict_readmission_risk(patient_data)
        treatment_patterns = self.analyze_treatment_patterns(patient_data)
        
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'risk_analysis': risk_analysis,
            'readmission_risk': readmission_risk,
            'treatment_patterns': treatment_patterns,
            'overall_assessment': self._generate_overall_assessment(risk_analysis, readmission_risk),
            'priority_actions': self._get_priority_actions(risk_analysis, readmission_risk)
        }
    
    def _generate_overall_assessment(self, risk_analysis: Dict, readmission_risk: Dict) -> str:
        """Generate overall patient assessment"""
        if risk_analysis['risk_level'] == 'High' or readmission_risk['risk_level'] == 'High':
            return 'High priority patient requiring immediate attention and close monitoring'
        elif risk_analysis['risk_level'] == 'Medium' or readmission_risk['risk_level'] == 'Medium':
            return 'Medium priority patient requiring regular monitoring and follow-up'
        else:
            return 'Low priority patient with standard care requirements'
    
    def _get_priority_actions(self, risk_analysis: Dict, readmission_risk: Dict) -> List[str]:
        """Get priority actions based on analysis"""
        actions = []
        
        if risk_analysis['risk_level'] == 'High':
            actions.extend([
                'Immediate clinical review',
                'Enhanced monitoring',
                'Specialist consultation if needed'
            ])
        
        if readmission_risk['risk_level'] == 'High':
            actions.extend([
                'Comprehensive discharge planning',
                'Patient education and support',
                'Follow-up scheduling'
            ])
        
        if not actions:
            actions.append('Continue standard care protocols')
        
        return actions
