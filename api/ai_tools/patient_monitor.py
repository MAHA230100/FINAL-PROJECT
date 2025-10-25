"""
Patient Monitor - AI tool for continuous patient monitoring and alerting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

class PatientMonitor:
    """AI-powered patient monitoring and alerting system"""
    
    def __init__(self):
        self.vital_thresholds = {
            'blood_pressure': {'systolic': {'high': 140, 'critical': 180}, 'diastolic': {'high': 90, 'critical': 110}},
            'heart_rate': {'high': 100, 'critical': 120, 'low': 60, 'critical_low': 50},
            'temperature': {'high': 100.4, 'critical': 102, 'low': 95, 'critical_low': 93},
            'oxygen_saturation': {'low': 95, 'critical': 90},
            'respiratory_rate': {'high': 20, 'critical': 25, 'low': 12, 'critical_low': 8}
        }
        
        self.alert_levels = {
            'critical': {'color': 'red', 'priority': 1, 'action': 'Immediate intervention required'},
            'high': {'color': 'orange', 'priority': 2, 'action': 'Urgent attention needed'},
            'medium': {'color': 'yellow', 'priority': 3, 'action': 'Monitor closely'},
            'low': {'color': 'green', 'priority': 4, 'action': 'Routine monitoring'}
        }
        
        self.monitoring_protocols = {
            'icu': {'frequency': 'continuous', 'parameters': ['all']},
            'step_down': {'frequency': 'every_2_hours', 'parameters': ['vitals', 'pain', 'consciousness']},
            'general': {'frequency': 'every_4_hours', 'parameters': ['vitals', 'pain']},
            'discharge': {'frequency': 'daily', 'parameters': ['vitals', 'symptoms']}
        }
    
    def analyze_vital_signs(self, vitals: Dict[str, float], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vital signs and generate alerts"""
        analysis = {
            'vitals': vitals,
            'alerts': [],
            'trends': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Check each vital sign against thresholds
        for vital_name, value in vitals.items():
            if vital_name in self.vital_thresholds:
                thresholds = self.vital_thresholds[vital_name]
                alert_level = self._determine_alert_level(vital_name, value, thresholds)
                
                if alert_level != 'normal':
                    analysis['alerts'].append({
                        'vital': vital_name,
                        'value': value,
                        'alert_level': alert_level,
                        'threshold': thresholds,
                        'message': self._get_alert_message(vital_name, value, alert_level)
                    })
        
        # Analyze trends (simplified - in real system would use historical data)
        analysis['trends'] = self._analyze_trends(vitals, patient_data)
        
        # Risk assessment
        analysis['risk_assessment'] = self._assess_risk(vitals, patient_data)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis['alerts'], analysis['risk_assessment'])
        
        return analysis
    
    def _determine_alert_level(self, vital_name: str, value: float, thresholds: Dict[str, Any]) -> str:
        """Determine alert level for a vital sign"""
        if vital_name == 'blood_pressure':
            # Handle blood pressure as systolic/diastolic
            if 'systolic' in str(value) and 'diastolic' in str(value):
                # Parse blood pressure string like "140/90"
                bp_parts = str(value).split('/')
                if len(bp_parts) == 2:
                    systolic = float(bp_parts[0])
                    diastolic = float(bp_parts[1])
                    
                    if systolic >= thresholds['systolic']['critical'] or diastolic >= thresholds['diastolic']['critical']:
                        return 'critical'
                    elif systolic >= thresholds['systolic']['high'] or diastolic >= thresholds['diastolic']['high']:
                        return 'high'
                    else:
                        return 'normal'
            else:
                # Single value - treat as systolic
                if value >= thresholds['systolic']['critical']:
                    return 'critical'
                elif value >= thresholds['systolic']['high']:
                    return 'high'
                else:
                    return 'normal'
        else:
            # Other vital signs
            if value >= thresholds.get('critical', float('inf')):
                return 'critical'
            elif value >= thresholds.get('high', float('inf')):
                return 'high'
            elif value <= thresholds.get('critical_low', float('-inf')):
                return 'critical'
            elif value <= thresholds.get('low', float('-inf')):
                return 'high'
            else:
                return 'normal'
    
    def _get_alert_message(self, vital_name: str, value: float, alert_level: str) -> str:
        """Generate alert message for vital sign"""
        messages = {
            'blood_pressure': {
                'critical': f'Critical hypertension: {value} - Immediate intervention required',
                'high': f'Elevated blood pressure: {value} - Monitor closely'
            },
            'heart_rate': {
                'critical': f'Critical heart rate: {value} - Immediate intervention required',
                'high': f'Elevated heart rate: {value} - Monitor closely'
            },
            'temperature': {
                'critical': f'Critical temperature: {value}°F - Immediate intervention required',
                'high': f'Elevated temperature: {value}°F - Monitor closely'
            },
            'oxygen_saturation': {
                'critical': f'Critical oxygen saturation: {value}% - Immediate oxygen therapy required',
                'high': f'Low oxygen saturation: {value}% - Monitor closely'
            }
        }
        
        return messages.get(vital_name, {}).get(alert_level, f'{vital_name}: {value} - {alert_level} alert')
    
    def _analyze_trends(self, vitals: Dict[str, float], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in vital signs"""
        trends = {
            'improving': [],
            'deteriorating': [],
            'stable': [],
            'concerning': []
        }
        
        # Simplified trend analysis (in real system would use historical data)
        for vital_name, value in vitals.items():
            if vital_name == 'blood_pressure':
                if 'systolic' in str(value):
                    bp_parts = str(value).split('/')
                    if len(bp_parts) == 2:
                        systolic = float(bp_parts[0])
                        if systolic > 160:
                            trends['concerning'].append(f'{vital_name}: {value}')
                        elif systolic < 100:
                            trends['concerning'].append(f'{vital_name}: {value}')
                        else:
                            trends['stable'].append(f'{vital_name}: {value}')
            elif vital_name == 'heart_rate':
                if value > 120:
                    trends['concerning'].append(f'{vital_name}: {value}')
                elif value < 50:
                    trends['concerning'].append(f'{vital_name}: {value}')
                else:
                    trends['stable'].append(f'{vital_name}: {value}')
            elif vital_name == 'temperature':
                if value > 102:
                    trends['concerning'].append(f'{vital_name}: {value}')
                elif value < 95:
                    trends['concerning'].append(f'{vital_name}: {value}')
                else:
                    trends['stable'].append(f'{vital_name}: {value}')
            elif vital_name == 'oxygen_saturation':
                if value < 90:
                    trends['concerning'].append(f'{vital_name}: {value}')
                else:
                    trends['stable'].append(f'{vital_name}: {value}')
        
        return trends
    
    def _assess_risk(self, vitals: Dict[str, float], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk based on vital signs and patient data"""
        risk_score = 0.0
        risk_factors = []
        
        # Age factor
        age = patient_data.get('age', 0)
        if age > 80:
            risk_score += 0.3
            risk_factors.append('Advanced age')
        elif age > 65:
            risk_score += 0.2
            risk_factors.append('Elderly')
        
        # Vital signs risk
        for vital_name, value in vitals.items():
            if vital_name == 'blood_pressure':
                if 'systolic' in str(value):
                    bp_parts = str(value).split('/')
                    if len(bp_parts) == 2:
                        systolic = float(bp_parts[0])
                        if systolic > 180:
                            risk_score += 0.4
                            risk_factors.append('Severe hypertension')
                        elif systolic > 140:
                            risk_score += 0.2
                            risk_factors.append('Hypertension')
            elif vital_name == 'heart_rate':
                if value > 120:
                    risk_score += 0.3
                    risk_factors.append('Tachycardia')
                elif value < 50:
                    risk_score += 0.3
                    risk_factors.append('Bradycardia')
            elif vital_name == 'temperature':
                if value > 102:
                    risk_score += 0.3
                    risk_factors.append('High fever')
                elif value < 95:
                    risk_score += 0.3
                    risk_factors.append('Hypothermia')
            elif vital_name == 'oxygen_saturation':
                if value < 90:
                    risk_score += 0.4
                    risk_factors.append('Severe hypoxemia')
                elif value < 95:
                    risk_score += 0.2
                    risk_factors.append('Mild hypoxemia')
        
        # Comorbidities risk
        comorbidities = patient_data.get('comorbidities', '').lower()
        if 'heart' in comorbidities:
            risk_score += 0.2
            risk_factors.append('Heart disease')
        if 'diabetes' in comorbidities:
            risk_score += 0.1
            risk_factors.append('Diabetes')
        if 'kidney' in comorbidities:
            risk_score += 0.2
            risk_factors.append('Kidney disease')
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'High'
        elif risk_score >= 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def _generate_recommendations(self, alerts: List[Dict], risk_assessment: Dict) -> List[str]:
        """Generate recommendations based on alerts and risk assessment"""
        recommendations = []
        
        # Recommendations based on alerts
        for alert in alerts:
            if alert['alert_level'] == 'critical':
                recommendations.append(f"Immediate intervention required for {alert['vital']}")
            elif alert['alert_level'] == 'high':
                recommendations.append(f"Close monitoring required for {alert['vital']}")
        
        # Recommendations based on risk level
        risk_level = risk_assessment.get('risk_level', 'Low')
        if risk_level == 'High':
            recommendations.extend([
                'Consider ICU admission',
                'Continuous monitoring required',
                'Physician notification immediately',
                'Family notification of high risk'
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                'Increased monitoring frequency',
                'Physician notification within 2 hours',
                'Consider step-down unit'
            ])
        else:
            recommendations.extend([
                'Routine monitoring',
                'Standard care protocols'
            ])
        
        return recommendations
    
    def generate_monitoring_plan(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive monitoring plan for patient"""
        monitoring_plan = {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'monitoring_level': 'general',
            'frequency': 'every_4_hours',
            'parameters': ['vitals', 'pain', 'consciousness'],
            'alerts': [],
            'protocols': [],
            'staff_assignments': [],
            'equipment_needed': []
        }
        
        # Determine monitoring level based on patient condition
        age = patient_data.get('age', 0)
        comorbidities = patient_data.get('comorbidities', '').lower()
        
        if any(condition in comorbidities for condition in ['icu', 'critical', 'ventilator']):
            monitoring_plan['monitoring_level'] = 'icu'
            monitoring_plan['frequency'] = 'continuous'
            monitoring_plan['parameters'] = ['all']
            monitoring_plan['equipment_needed'] = [
                'Continuous vital sign monitor',
                'Oxygen saturation monitor',
                'Cardiac monitor',
                'Ventilator (if needed)'
            ]
        elif age > 75 or any(condition in comorbidities for condition in ['heart', 'diabetes', 'kidney']):
            monitoring_plan['monitoring_level'] = 'step_down'
            monitoring_plan['frequency'] = 'every_2_hours'
            monitoring_plan['parameters'] = ['vitals', 'pain', 'consciousness', 'mobility']
            monitoring_plan['equipment_needed'] = [
                'Vital sign monitor',
                'Oxygen saturation monitor',
                'Cardiac monitor'
            ]
        else:
            monitoring_plan['monitoring_level'] = 'general'
            monitoring_plan['frequency'] = 'every_4_hours'
            monitoring_plan['parameters'] = ['vitals', 'pain']
            monitoring_plan['equipment_needed'] = [
                'Vital sign monitor',
                'Oxygen saturation monitor'
            ]
        
        # Set up alerts
        monitoring_plan['alerts'] = [
            'Critical vital signs',
            'Pain score > 7',
            'Change in consciousness',
            'Fall risk',
            'Medication reactions'
        ]
        
        # Set up protocols
        monitoring_plan['protocols'] = [
            'Vital signs assessment',
            'Pain assessment',
            'Neurological assessment',
            'Mobility assessment',
            'Medication administration',
            'Patient education'
        ]
        
        # Staff assignments
        if monitoring_plan['monitoring_level'] == 'icu':
            monitoring_plan['staff_assignments'] = [
                'ICU nurse (1:1 ratio)',
                'Respiratory therapist',
                'ICU physician',
                'Pharmacist consultation'
            ]
        elif monitoring_plan['monitoring_level'] == 'step_down':
            monitoring_plan['staff_assignments'] = [
                'Step-down nurse (1:2 ratio)',
                'Respiratory therapist',
                'Physician rounds'
            ]
        else:
            monitoring_plan['staff_assignments'] = [
                'General nurse (1:4 ratio)',
                'Physician rounds'
            ]
        
        return monitoring_plan
    
    def check_medication_effects(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for medication effects and side effects"""
        medication_effects = {
            'medications': patient_data.get('medications', []),
            'side_effects': [],
            'interactions': [],
            'monitoring_requirements': [],
            'recommendations': []
        }
        
        # Check for common side effects
        medications = patient_data.get('medications', [])
        for med in medications:
            med_lower = med.lower()
            if 'warfarin' in med_lower:
                medication_effects['side_effects'].append({
                    'medication': med,
                    'side_effect': 'Bleeding risk',
                    'severity': 'High',
                    'monitoring': 'INR, bleeding signs'
                })
                medication_effects['monitoring_requirements'].append('INR every 1-2 weeks')
            elif 'digoxin' in med_lower:
                medication_effects['side_effects'].append({
                    'medication': med,
                    'side_effect': 'Digoxin toxicity',
                    'severity': 'High',
                    'monitoring': 'Digoxin level, symptoms'
                })
                medication_effects['monitoring_requirements'].append('Digoxin level every 3-6 months')
            elif 'insulin' in med_lower:
                medication_effects['side_effects'].append({
                    'medication': med,
                    'side_effect': 'Hypoglycemia',
                    'severity': 'High',
                    'monitoring': 'Blood glucose, symptoms'
                })
                medication_effects['monitoring_requirements'].append('Blood glucose monitoring')
            elif 'ace' in med_lower:
                medication_effects['side_effects'].append({
                    'medication': med,
                    'side_effect': 'Hyperkalemia, cough',
                    'severity': 'Medium',
                    'monitoring': 'Potassium level, symptoms'
                })
                medication_effects['monitoring_requirements'].append('Potassium level every 3-6 months')
        
        # Check for drug interactions
        for i, med1 in enumerate(medications):
            for j, med2 in enumerate(medications[i+1:], i+1):
                med1_lower = med1.lower()
                med2_lower = med2.lower()
                
                if 'warfarin' in med1_lower and 'aspirin' in med2_lower:
                    medication_effects['interactions'].append({
                        'medication1': med1,
                        'medication2': med2,
                        'interaction': 'Increased bleeding risk',
                        'severity': 'High'
                    })
                elif 'digoxin' in med1_lower and 'furosemide' in med2_lower:
                    medication_effects['interactions'].append({
                        'medication1': med1,
                        'medication2': med2,
                        'interaction': 'Digoxin toxicity risk',
                        'severity': 'Medium'
                    })
        
        # Generate recommendations
        if medication_effects['side_effects']:
            medication_effects['recommendations'].append('Monitor for side effects closely')
        if medication_effects['interactions']:
            medication_effects['recommendations'].append('Review drug interactions with pharmacist')
        if medication_effects['monitoring_requirements']:
            medication_effects['recommendations'].append('Ensure monitoring requirements are met')
        
        return medication_effects
    
    def generate_patient_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive patient monitoring summary"""
        vitals = {
            'blood_pressure': patient_data.get('vitals_bp', 0),
            'heart_rate': patient_data.get('vitals_hr', 0),
            'temperature': patient_data.get('temperature', 98.6),
            'oxygen_saturation': patient_data.get('oxygen_saturation', 98),
            'respiratory_rate': patient_data.get('respiratory_rate', 16)
        }
        
        # Analyze vital signs
        vital_analysis = self.analyze_vital_signs(vitals, patient_data)
        
        # Generate monitoring plan
        monitoring_plan = self.generate_monitoring_plan(patient_data)
        
        # Check medication effects
        medication_effects = self.check_medication_effects(patient_data)
        
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'vital_analysis': vital_analysis,
            'monitoring_plan': monitoring_plan,
            'medication_effects': medication_effects,
            'summary_date': datetime.now().isoformat(),
            'overall_status': self._determine_overall_status(vital_analysis, monitoring_plan),
            'next_assessment': self._calculate_next_assessment(monitoring_plan)
        }
    
    def _determine_overall_status(self, vital_analysis: Dict, monitoring_plan: Dict) -> str:
        """Determine overall patient status"""
        alerts = vital_analysis.get('alerts', [])
        risk_level = vital_analysis.get('risk_assessment', {}).get('risk_level', 'Low')
        
        if any(alert['alert_level'] == 'critical' for alert in alerts):
            return 'Critical'
        elif any(alert['alert_level'] == 'high' for alert in alerts):
            return 'High Risk'
        elif risk_level == 'High':
            return 'High Risk'
        elif risk_level == 'Medium':
            return 'Medium Risk'
        else:
            return 'Stable'
    
    def _calculate_next_assessment(self, monitoring_plan: Dict) -> str:
        """Calculate next assessment time"""
        frequency = monitoring_plan.get('frequency', 'every_4_hours')
        
        if frequency == 'continuous':
            return 'Continuous monitoring'
        elif frequency == 'every_2_hours':
            next_time = datetime.now() + timedelta(hours=2)
            return next_time.strftime('%Y-%m-%d %H:%M')
        elif frequency == 'every_4_hours':
            next_time = datetime.now() + timedelta(hours=4)
            return next_time.strftime('%Y-%m-%d %H:%M')
        else:
            next_time = datetime.now() + timedelta(hours=4)
            return next_time.strftime('%Y-%m-%d %H:%M')
