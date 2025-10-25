"""
Risk Assessor - AI tool for healthcare risk assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class RiskAssessor:
    """AI-powered risk assessment tool for healthcare"""
    
    def __init__(self):
        self.risk_models = {
            'mortality': {
                'factors': ['age', 'comorbidities', 'vital_signs', 'lab_values'],
                'weights': {'age': 0.3, 'comorbidities': 0.4, 'vital_signs': 0.2, 'lab_values': 0.1}
            },
            'readmission': {
                'factors': ['length_of_stay', 'discharge_disposition', 'medications', 'follow_up'],
                'weights': {'length_of_stay': 0.3, 'discharge_disposition': 0.3, 'medications': 0.2, 'follow_up': 0.2}
            },
            'infection': {
                'factors': ['immune_status', 'procedures', 'environment', 'medications'],
                'weights': {'immune_status': 0.4, 'procedures': 0.3, 'environment': 0.2, 'medications': 0.1}
            }
        }
        
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def calculate_mortality_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate mortality risk score"""
        risk_score = 0.0
        risk_factors = []
        
        # Age factor
        age = patient_data.get('age', 0)
        if age > 80:
            age_risk = 0.8
            risk_factors.append('Advanced age (>80)')
        elif age > 65:
            age_risk = 0.5
            risk_factors.append('Elderly (>65)')
        else:
            age_risk = 0.1
        
        risk_score += age_risk * self.risk_models['mortality']['weights']['age']
        
        # Comorbidities factor
        comorbidities = patient_data.get('comorbidities', '')
        comorbidity_risk = 0.0
        if comorbidities:
            comorbidity_list = comorbidities.split(',')
            for comorbidity in comorbidity_list:
                comorbidity = comorbidity.strip().lower()
                if any(condition in comorbidity for condition in ['heart', 'diabetes', 'cancer', 'kidney']):
                    comorbidity_risk += 0.3
                    risk_factors.append(f'High-risk comorbidity: {comorbidity}')
                elif any(condition in comorbidity for condition in ['hypertension', 'asthma', 'copd']):
                    comorbidity_risk += 0.2
                    risk_factors.append(f'Moderate-risk comorbidity: {comorbidity}')
        
        risk_score += min(comorbidity_risk, 1.0) * self.risk_models['mortality']['weights']['comorbidities']
        
        # Vital signs factor
        vital_risk = 0.0
        bp = patient_data.get('vitals_bp', 0)
        hr = patient_data.get('vitals_hr', 0)
        temp = patient_data.get('temperature', 98.6)
        spo2 = patient_data.get('oxygen_saturation', 98)
        
        if bp > 180 or bp < 90:
            vital_risk += 0.4
            risk_factors.append('Abnormal blood pressure')
        if hr > 120 or hr < 50:
            vital_risk += 0.3
            risk_factors.append('Abnormal heart rate')
        if temp > 102 or temp < 96:
            vital_risk += 0.3
            risk_factors.append('Abnormal temperature')
        if spo2 < 90:
            vital_risk += 0.5
            risk_factors.append('Low oxygen saturation')
        
        risk_score += min(vital_risk, 1.0) * self.risk_models['mortality']['weights']['vital_signs']
        
        # Lab values factor
        lab_risk = 0.0
        if patient_data.get('lab_values'):
            lab_values = patient_data['lab_values']
            if lab_values.get('creatinine', 0) > 2.0:
                lab_risk += 0.3
                risk_factors.append('Elevated creatinine')
            if lab_values.get('glucose', 0) > 200:
                lab_risk += 0.2
                risk_factors.append('Hyperglycemia')
            if lab_values.get('wbc', 0) > 15000:
                lab_risk += 0.2
                risk_factors.append('Leukocytosis')
        
        risk_score += min(lab_risk, 1.0) * self.risk_models['mortality']['weights']['lab_values']
        
        # Determine risk level
        if risk_score >= self.risk_thresholds['high']:
            risk_level = 'High'
        elif risk_score >= self.risk_thresholds['medium']:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._get_mortality_recommendations(risk_level, risk_factors)
        }
    
    def calculate_readmission_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate readmission risk score"""
        risk_score = 0.0
        risk_factors = []
        
        # Length of stay factor
        los = patient_data.get('length_of_stay', 0)
        if los > 7:
            los_risk = 0.6
            risk_factors.append('Extended length of stay')
        elif los > 3:
            los_risk = 0.3
            risk_factors.append('Moderate length of stay')
        else:
            los_risk = 0.1
        
        risk_score += los_risk * self.risk_models['readmission']['weights']['length_of_stay']
        
        # Discharge disposition factor
        discharge_disp = patient_data.get('discharge_disposition', '').lower()
        if 'home' not in discharge_disp:
            disp_risk = 0.7
            risk_factors.append('Non-home discharge')
        else:
            disp_risk = 0.2
            risk_factors.append('Home discharge')
        
        risk_score += disp_risk * self.risk_models['readmission']['weights']['discharge_disposition']
        
        # Medications factor
        medications = patient_data.get('medications', [])
        med_risk = 0.0
        if len(medications) > 5:
            med_risk += 0.4
            risk_factors.append('Polypharmacy')
        if any('insulin' in med.lower() for med in medications):
            med_risk += 0.3
            risk_factors.append('Insulin-dependent diabetes')
        if any('warfarin' in med.lower() for med in medications):
            med_risk += 0.2
            risk_factors.append('Anticoagulation therapy')
        
        risk_score += min(med_risk, 1.0) * self.risk_models['readmission']['weights']['medications']
        
        # Follow-up factor
        follow_up = patient_data.get('follow_up_scheduled', False)
        if not follow_up:
            follow_risk = 0.5
            risk_factors.append('No follow-up scheduled')
        else:
            follow_risk = 0.1
            risk_factors.append('Follow-up scheduled')
        
        risk_score += follow_risk * self.risk_models['readmission']['weights']['follow_up']
        
        # Determine risk level
        if risk_score >= self.risk_thresholds['high']:
            risk_level = 'High'
        elif risk_score >= self.risk_thresholds['medium']:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._get_readmission_recommendations(risk_level, risk_factors)
        }
    
    def calculate_infection_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate infection risk score"""
        risk_score = 0.0
        risk_factors = []
        
        # Immune status factor
        immune_risk = 0.0
        if patient_data.get('immunocompromised', False):
            immune_risk += 0.6
            risk_factors.append('Immunocompromised status')
        if patient_data.get('diabetes', False):
            immune_risk += 0.3
            risk_factors.append('Diabetes (immune compromise)')
        if patient_data.get('age', 0) > 65:
            immune_risk += 0.2
            risk_factors.append('Age-related immune decline')
        
        risk_score += min(immune_risk, 1.0) * self.risk_models['infection']['weights']['immune_status']
        
        # Procedures factor
        procedures = patient_data.get('procedures', [])
        proc_risk = 0.0
        if any('surgery' in proc.lower() for proc in procedures):
            proc_risk += 0.5
            risk_factors.append('Surgical procedure')
        if any('catheter' in proc.lower() for proc in procedures):
            proc_risk += 0.4
            risk_factors.append('Catheter placement')
        if any('intubation' in proc.lower() for proc in procedures):
            proc_risk += 0.3
            risk_factors.append('Intubation')
        
        risk_score += min(proc_risk, 1.0) * self.risk_models['infection']['weights']['procedures']
        
        # Environment factor
        env_risk = 0.0
        if patient_data.get('icu_stay', False):
            env_risk += 0.4
            risk_factors.append('ICU environment')
        if patient_data.get('isolation', False):
            env_risk += 0.3
            risk_factors.append('Isolation precautions')
        if patient_data.get('ventilator', False):
            env_risk += 0.5
            risk_factors.append('Mechanical ventilation')
        
        risk_score += min(env_risk, 1.0) * self.risk_models['infection']['weights']['environment']
        
        # Medications factor
        medications = patient_data.get('medications', [])
        med_risk = 0.0
        if any('steroid' in med.lower() for med in medications):
            med_risk += 0.3
            risk_factors.append('Steroid therapy')
        if any('immunosuppressant' in med.lower() for med in medications):
            med_risk += 0.4
            risk_factors.append('Immunosuppressive therapy')
        if any('antibiotic' in med.lower() for med in medications):
            med_risk += 0.2
            risk_factors.append('Antibiotic therapy')
        
        risk_score += min(med_risk, 1.0) * self.risk_models['infection']['weights']['medications']
        
        # Determine risk level
        if risk_score >= self.risk_thresholds['high']:
            risk_level = 'High'
        elif risk_score >= self.risk_thresholds['medium']:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._get_infection_recommendations(risk_level, risk_factors)
        }
    
    def _get_mortality_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Get mortality risk recommendations"""
        recommendations = []
        
        if risk_level == 'High':
            recommendations.extend([
                'Immediate intensive monitoring required',
                'Consider ICU admission',
                'Family notification of high-risk status',
                'Palliative care consultation if appropriate'
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                'Close monitoring every 4 hours',
                'Regular vital sign assessment',
                'Consider step-down unit',
                'Family education about condition'
            ])
        else:
            recommendations.extend([
                'Routine monitoring',
                'Standard care protocols',
                'Patient education',
                'Regular reassessment'
            ])
        
        return recommendations
    
    def _get_readmission_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Get readmission risk recommendations"""
        recommendations = []
        
        if risk_level == 'High':
            recommendations.extend([
                'Enhanced discharge planning',
                'Medication reconciliation',
                'Follow-up appointment within 48 hours',
                'Home health services if needed',
                'Patient education reinforcement'
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                'Standard discharge planning',
                'Follow-up appointment within 1 week',
                'Medication review',
                'Patient education'
            ])
        else:
            recommendations.extend([
                'Routine discharge planning',
                'Standard follow-up',
                'Patient education'
            ])
        
        return recommendations
    
    def _get_infection_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Get infection risk recommendations"""
        recommendations = []
        
        if risk_level == 'High':
            recommendations.extend([
                'Strict infection control measures',
                'Antimicrobial prophylaxis if indicated',
                'Regular infection surveillance',
                'Isolation precautions if needed',
                'Staff education on infection prevention'
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                'Standard infection control',
                'Regular monitoring for signs of infection',
                'Hand hygiene education',
                'Environmental cleaning protocols'
            ])
        else:
            recommendations.extend([
                'Basic infection control',
                'Patient education on hygiene',
                'Standard monitoring'
            ])
        
        return recommendations
    
    def generate_risk_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment summary"""
        mortality_risk = self.calculate_mortality_risk(patient_data)
        readmission_risk = self.calculate_readmission_risk(patient_data)
        infection_risk = self.calculate_infection_risk(patient_data)
        
        # Overall risk assessment
        overall_risk_score = (mortality_risk['risk_score'] + 
                            readmission_risk['risk_score'] + 
                            infection_risk['risk_score']) / 3
        
        if overall_risk_score >= self.risk_thresholds['high']:
            overall_risk_level = 'High'
        elif overall_risk_score >= self.risk_thresholds['medium']:
            overall_risk_level = 'Medium'
        else:
            overall_risk_level = 'Low'
        
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'overall_risk_score': round(overall_risk_score, 3),
            'overall_risk_level': overall_risk_level,
            'mortality_risk': mortality_risk,
            'readmission_risk': readmission_risk,
            'infection_risk': infection_risk,
            'assessment_date': datetime.now().isoformat(),
            'recommendations': {
                'immediate_actions': self._get_immediate_actions(overall_risk_level),
                'monitoring_plan': self._get_monitoring_plan(overall_risk_level),
                'follow_up_plan': self._get_follow_up_plan(overall_risk_level)
            }
        }
    
    def _get_immediate_actions(self, risk_level: str) -> List[str]:
        """Get immediate actions based on risk level"""
        if risk_level == 'High':
            return [
                'Immediate physician notification',
                'Enhanced monitoring setup',
                'Family notification',
                'Consider ICU consultation'
            ]
        elif risk_level == 'Medium':
            return [
                'Physician notification within 2 hours',
                'Increased monitoring frequency',
                'Review care plan'
            ]
        else:
            return [
                'Routine monitoring',
                'Standard care protocols'
            ]
    
    def _get_monitoring_plan(self, risk_level: str) -> List[str]:
        """Get monitoring plan based on risk level"""
        if risk_level == 'High':
            return [
                'Continuous vital sign monitoring',
                'Hourly assessments',
                'Daily physician rounds',
                'Specialist consultation'
            ]
        elif risk_level == 'Medium':
            return [
                'Vital signs every 4 hours',
                'Daily assessments',
                'Regular physician rounds'
            ]
        else:
            return [
                'Standard vital sign monitoring',
                'Routine assessments'
            ]
    
    def _get_follow_up_plan(self, risk_level: str) -> List[str]:
        """Get follow-up plan based on risk level"""
        if risk_level == 'High':
            return [
                'Daily physician rounds',
                'Specialist consultation',
                'Family meetings',
                'Care plan review'
            ]
        elif risk_level == 'Medium':
            return [
                'Every 2-3 day physician rounds',
                'Regular assessments',
                'Care plan updates'
            ]
        else:
            return [
                'Standard follow-up',
                'Routine assessments'
            ]
