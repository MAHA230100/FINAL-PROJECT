"""
Medication Advisor - AI tool for medication management and recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class MedicationAdvisor:
    """AI-powered medication advisor for healthcare"""
    
    def __init__(self):
        self.medication_database = {
            'antihypertensives': {
                'ace_inhibitors': ['lisinopril', 'enalapril', 'ramipril'],
                'arbs': ['losartan', 'valsartan', 'candesartan'],
                'beta_blockers': ['metoprolol', 'atenolol', 'propranolol'],
                'calcium_channel_blockers': ['amlodipine', 'diltiazem', 'verapamil'],
                'diuretics': ['hydrochlorothiazide', 'furosemide', 'spironolactone']
            },
            'diabetes_medications': {
                'insulins': ['insulin_glargine', 'insulin_lispro', 'insulin_aspart'],
                'oral_agents': ['metformin', 'glipizide', 'pioglitazone'],
                'injectables': ['liraglutide', 'dulaglutide', 'semaglutide']
            },
            'cardiac_medications': {
                'antiplatelets': ['aspirin', 'clopidogrel', 'prasugrel'],
                'anticoagulants': ['warfarin', 'apixaban', 'rivaroxaban'],
                'statins': ['atorvastatin', 'simvastatin', 'rosuvastatin']
            }
        }
        
        self.drug_interactions = {
            ('warfarin', 'aspirin'): {
                'severity': 'High',
                'effect': 'Increased bleeding risk',
                'recommendation': 'Monitor INR closely, consider alternative'
            },
            ('warfarin', 'ibuprofen'): {
                'severity': 'High',
                'effect': 'Increased bleeding risk',
                'recommendation': 'Avoid NSAIDs, use acetaminophen'
            },
            ('digoxin', 'furosemide'): {
                'severity': 'Medium',
                'effect': 'Digoxin toxicity risk',
                'recommendation': 'Monitor digoxin levels, adjust dose'
            },
            ('ace_inhibitor', 'potassium'): {
                'severity': 'Medium',
                'effect': 'Hyperkalemia risk',
                'recommendation': 'Monitor potassium levels'
            },
            ('metformin', 'contrast_dye'): {
                'severity': 'High',
                'effect': 'Lactic acidosis risk',
                'recommendation': 'Hold metformin before contrast'
            }
        }
        
        self.contraindications = {
            'pregnancy': ['warfarin', 'ace_inhibitors', 'arbs', 'statins'],
            'liver_disease': ['statins', 'metformin', 'warfarin'],
            'kidney_disease': ['metformin', 'ace_inhibitors', 'arbs'],
            'heart_failure': ['beta_blockers', 'calcium_channel_blockers']
        }
    
    def analyze_medication_list(self, medications: List[str], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current medication list for issues"""
        analysis = {
            'medications': medications,
            'interactions': [],
            'contraindications': [],
            'duplications': [],
            'recommendations': [],
            'safety_score': 0.0
        }
        
        # Check for drug interactions
        for i, med1 in enumerate(medications):
            for j, med2 in enumerate(medications[i+1:], i+1):
                med1_lower = med1.lower()
                med2_lower = med2.lower()
                
                for (drug1, drug2), interaction_info in self.drug_interactions.items():
                    if (drug1 in med1_lower and drug2 in med2_lower) or \
                       (drug2 in med1_lower and drug1 in med2_lower):
                        analysis['interactions'].append({
                            'medication1': med1,
                            'medication2': med2,
                            'interaction': interaction_info
                        })
        
        # Check for contraindications
        patient_conditions = patient_data.get('comorbidities', '').lower()
        for condition, contraindicated_drugs in self.contraindications.items():
            if condition in patient_conditions:
                for med in medications:
                    med_lower = med.lower()
                    for contraindicated_drug in contraindicated_drugs:
                        if contraindicated_drug in med_lower:
                            analysis['contraindications'].append({
                                'medication': med,
                                'condition': condition,
                                'risk': 'High'
                            })
        
        # Check for duplications
        medication_classes = {}
        for med in medications:
            med_lower = med.lower()
            for drug_class, drugs in self.medication_database.items():
                for subclass, drug_list in drugs.items():
                    for drug in drug_list:
                        if drug in med_lower:
                            if drug_class not in medication_classes:
                                medication_classes[drug_class] = []
                            medication_classes[drug_class].append(med)
        
        for drug_class, meds in medication_classes.items():
            if len(meds) > 1:
                analysis['duplications'].append({
                    'drug_class': drug_class,
                    'medications': meds,
                    'recommendation': 'Consider consolidating to single agent'
                })
        
        # Calculate safety score
        total_issues = len(analysis['interactions']) + len(analysis['contraindications']) + len(analysis['duplications'])
        analysis['safety_score'] = max(0, 1.0 - (total_issues * 0.2))
        
        # Generate recommendations
        if analysis['interactions']:
            analysis['recommendations'].append('Review drug interactions with pharmacist')
        if analysis['contraindications']:
            analysis['recommendations'].append('Consider alternative medications for contraindicated drugs')
        if analysis['duplications']:
            analysis['recommendations'].append('Consolidate duplicate drug classes')
        
        return analysis
    
    def recommend_medication_changes(self, current_medications: List[str], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend medication changes based on patient data"""
        recommendations = {
            'current_medications': current_medications,
            'recommended_changes': [],
            'new_medications': [],
            'discontinued_medications': [],
            'dose_adjustments': [],
            'monitoring_requirements': []
        }
        
        # Analyze current medications
        analysis = self.analyze_medication_list(current_medications, patient_data)
        
        # Recommend discontinuation of problematic medications
        for interaction in analysis['interactions']:
            if interaction['interaction']['severity'] == 'High':
                recommendations['discontinued_medications'].append({
                    'medication': interaction['medication1'],
                    'reason': f"High-risk interaction with {interaction['medication2']}",
                    'alternative': 'Consider alternative medication'
                })
        
        for contraindication in analysis['contraindications']:
            recommendations['discontinued_medications'].append({
                'medication': contraindication['medication'],
                'reason': f"Contraindicated in {contraindication['condition']}",
                'alternative': 'Consider alternative medication'
            })
        
        # Recommend new medications based on conditions
        conditions = patient_data.get('comorbidities', '').lower()
        
        if 'hypertension' in conditions:
            if not any('ace' in med.lower() or 'arb' in med.lower() for med in current_medications):
                recommendations['new_medications'].append({
                    'medication': 'ACE inhibitor or ARB',
                    'indication': 'Hypertension management',
                    'examples': ['Lisinopril', 'Losartan']
                })
        
        if 'diabetes' in conditions:
            if not any('metformin' in med.lower() for med in current_medications):
                recommendations['new_medications'].append({
                    'medication': 'Metformin',
                    'indication': 'Type 2 diabetes management',
                    'note': 'First-line therapy for type 2 diabetes'
                })
        
        if 'heart_failure' in conditions:
            if not any('ace' in med.lower() for med in current_medications):
                recommendations['new_medications'].append({
                    'medication': 'ACE inhibitor',
                    'indication': 'Heart failure management',
                    'examples': ['Lisinopril', 'Enalapril']
                })
        
        # Recommend dose adjustments based on age and kidney function
        age = patient_data.get('age', 0)
        if age > 65:
            for med in current_medications:
                if 'warfarin' in med.lower():
                    recommendations['dose_adjustments'].append({
                        'medication': med,
                        'adjustment': 'Consider lower starting dose',
                        'reason': 'Age-related increased sensitivity'
                    })
        
        # Set monitoring requirements
        for med in current_medications:
            med_lower = med.lower()
            if 'warfarin' in med_lower:
                recommendations['monitoring_requirements'].append({
                    'medication': med,
                    'monitoring': 'INR every 1-2 weeks',
                    'target': 'INR 2-3'
                })
            elif 'digoxin' in med_lower:
                recommendations['monitoring_requirements'].append({
                    'medication': med,
                    'monitoring': 'Digoxin level every 3-6 months',
                    'target': '0.5-2.0 ng/mL'
                })
            elif 'lithium' in med_lower:
                recommendations['monitoring_requirements'].append({
                    'medication': med,
                    'monitoring': 'Lithium level every 3-6 months',
                    'target': '0.6-1.2 mEq/L'
                })
        
        return recommendations
    
    def optimize_medication_regimen(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize medication regimen for patient"""
        optimization = {
            'current_regimen': patient_data.get('medications', []),
            'optimized_regimen': [],
            'rationale': [],
            'expected_benefits': [],
            'monitoring_plan': []
        }
        
        # Start with current medications
        optimized_meds = patient_data.get('medications', []).copy()
        
        # Remove problematic medications
        analysis = self.analyze_medication_list(optimized_meds, patient_data)
        for interaction in analysis['interactions']:
            if interaction['interaction']['severity'] == 'High':
                if interaction['medication1'] in optimized_meds:
                    optimized_meds.remove(interaction['medication1'])
                    optimization['rationale'].append(f"Removed {interaction['medication1']} due to high-risk interaction")
        
        # Add evidence-based medications
        conditions = patient_data.get('comorbidities', '').lower()
        
        if 'hypertension' in conditions:
            if not any('ace' in med.lower() for med in optimized_meds):
                optimized_meds.append('ACE inhibitor')
                optimization['rationale'].append('Added ACE inhibitor for hypertension management')
        
        if 'diabetes' in conditions:
            if not any('metformin' in med.lower() for med in optimized_meds):
                optimized_meds.append('Metformin')
                optimization['rationale'].append('Added Metformin as first-line diabetes therapy')
        
        if 'heart_failure' in conditions:
            if not any('ace' in med.lower() for med in optimized_meds):
                optimized_meds.append('ACE inhibitor')
                optimization['rationale'].append('Added ACE inhibitor for heart failure management')
            if not any('beta' in med.lower() for med in optimized_meds):
                optimized_meds.append('Beta-blocker')
                optimization['rationale'].append('Added Beta-blocker for heart failure management')
        
        # Add statin if cardiovascular risk
        if any(condition in conditions for condition in ['heart', 'diabetes', 'hypertension']):
            if not any('statin' in med.lower() for med in optimized_meds):
                optimized_meds.append('Statin')
                optimization['rationale'].append('Added Statin for cardiovascular risk reduction')
        
        optimization['optimized_regimen'] = optimized_meds
        
        # Expected benefits
        optimization['expected_benefits'] = [
            'Improved blood pressure control',
            'Better diabetes management',
            'Reduced cardiovascular risk',
            'Fewer drug interactions',
            'Simplified medication regimen'
        ]
        
        # Monitoring plan
        optimization['monitoring_plan'] = [
            'Regular blood pressure monitoring',
            'HbA1c every 3 months if diabetic',
            'Lipid panel annually',
            'Kidney function every 6 months',
            'Medication adherence assessment'
        ]
        
        return optimization
    
    def check_medication_adherence(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check medication adherence and provide recommendations"""
        adherence_assessment = {
            'adherence_score': 0.0,
            'risk_factors': [],
            'recommendations': [],
            'interventions': []
        }
        
        # Factors affecting adherence
        age = patient_data.get('age', 0)
        if age > 75:
            adherence_assessment['risk_factors'].append('Advanced age')
            adherence_assessment['adherence_score'] -= 0.2
        
        # Number of medications
        med_count = len(patient_data.get('medications', []))
        if med_count > 5:
            adherence_assessment['risk_factors'].append('Polypharmacy (>5 medications)')
            adherence_assessment['adherence_score'] -= 0.3
        elif med_count > 3:
            adherence_assessment['risk_factors'].append('Multiple medications')
            adherence_assessment['adherence_score'] -= 0.1
        
        # Complex dosing regimens
        complex_dosing = any('twice' in med.lower() or 'three' in med.lower() for med in patient_data.get('medications', []))
        if complex_dosing:
            adherence_assessment['risk_factors'].append('Complex dosing regimens')
            adherence_assessment['adherence_score'] -= 0.2
        
        # Cognitive function
        if patient_data.get('cognitive_impairment', False):
            adherence_assessment['risk_factors'].append('Cognitive impairment')
            adherence_assessment['adherence_score'] -= 0.4
        
        # Social support
        if not patient_data.get('caregiver_support', False):
            adherence_assessment['risk_factors'].append('Limited social support')
            adherence_assessment['adherence_score'] -= 0.2
        
        # Calculate final adherence score
        adherence_assessment['adherence_score'] = max(0, min(1, adherence_assessment['adherence_score'] + 1))
        
        # Generate recommendations
        if adherence_assessment['adherence_score'] < 0.5:
            adherence_assessment['recommendations'].extend([
                'High-risk for non-adherence',
                'Consider medication simplification',
                'Implement adherence monitoring',
                'Provide patient education'
            ])
            adherence_assessment['interventions'].extend([
                'Pill organizer setup',
                'Medication reminder system',
                'Caregiver involvement',
                'Regular follow-up calls'
            ])
        elif adherence_assessment['adherence_score'] < 0.7:
            adherence_assessment['recommendations'].extend([
                'Moderate risk for non-adherence',
                'Monitor adherence closely',
                'Provide education and support'
            ])
            adherence_assessment['interventions'].extend([
                'Patient education',
                'Adherence monitoring',
                'Regular follow-up'
            ])
        else:
            adherence_assessment['recommendations'].extend([
                'Good adherence expected',
                'Continue current support',
                'Regular monitoring'
            ])
        
        return adherence_assessment
    
    def generate_medication_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive medication summary"""
        current_medications = patient_data.get('medications', [])
        
        # Analyze current medications
        medication_analysis = self.analyze_medication_list(current_medications, patient_data)
        
        # Get recommendations
        change_recommendations = self.recommend_medication_changes(current_medications, patient_data)
        
        # Optimize regimen
        optimized_regimen = self.optimize_medication_regimen(patient_data)
        
        # Check adherence
        adherence_assessment = self.check_medication_adherence(patient_data)
        
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'current_medications': current_medications,
            'medication_analysis': medication_analysis,
            'change_recommendations': change_recommendations,
            'optimized_regimen': optimized_regimen,
            'adherence_assessment': adherence_assessment,
            'summary_date': datetime.now().isoformat(),
            'overall_recommendations': [
                'Regular medication review',
                'Patient education on medications',
                'Monitor for side effects',
                'Assess adherence regularly'
            ]
        }
