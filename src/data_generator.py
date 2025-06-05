"""
Synthetic EHR Data Generator for Patient Readmission Prediction
Creates realistic hospital data for ML training while maintaining privacy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, List, Dict

class SyntheticEHRGenerator:
    def __init__(self, n_patients: int = 100000, random_state: int = 42):
        """
        Initialize the synthetic EHR data generator
        
        Args:
            n_patients: Number of unique patients to generate
            random_state: Random seed for reproducibility
        """
        self.n_patients = n_patients
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define realistic medical data
        self.icd_codes = {
            'I50.9': 'Heart failure, unspecified',
            'E11.9': 'Type 2 diabetes mellitus without complications',
            'I10': 'Essential hypertension',
            'J44.1': 'Chronic obstructive pulmonary disease with acute exacerbation',
            'N18.6': 'End stage renal disease',
            'F03.90': 'Unspecified dementia without behavioral disturbance',
            'I25.10': 'Atherosclerotic heart disease of native coronary artery',
            'J18.9': 'Pneumonia, unspecified organism',
            'R06.02': 'Shortness of breath',
            'R50.9': 'Fever, unspecified'
        }
        
        self.medications = [
            'Metformin', 'Lisinopril', 'Atorvastatin', 'Metoprolol', 'Amlodipine',
            'Omeprazole', 'Levothyroxine', 'Albuterol', 'Furosemide', 'Warfarin',
            'Insulin', 'Prednisone', 'Gabapentin', 'Tramadol', 'Sertraline'
        ]
        
        self.procedures = {
            '99213': 'Office visit, established patient',
            '93000': 'Electrocardiogram',
            '80053': 'Comprehensive metabolic panel',
            '85025': 'Complete blood count',
            '36415': 'Blood draw',
            '71020': 'Chest X-ray',
            '93005': 'Electrocardiogram interpretation',
            '99291': 'Critical care',
            '45378': 'Colonoscopy',
            '47562': 'Laparoscopic cholecystectomy'
        }

    def generate_patients(self) -> pd.DataFrame:
        """Generate patient demographic data"""
        patients = []
        
        for patient_id in range(1, self.n_patients + 1):
            # Age distribution realistic for hospital admissions
            age = np.random.choice(
                range(18, 90), 
                p=self._age_distribution()
            )
            
            gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
            
            # Insurance type affects readmission risk
            insurance = np.random.choice(
                ['Medicare', 'Medicaid', 'Private', 'Uninsured'], 
                p=[0.35, 0.25, 0.35, 0.05]
            )
            
            # Socioeconomic factors
            zip_code = f"{np.random.randint(10000, 99999)}"
            
            patients.append({
                'patient_id': patient_id,
                'age': age,
                'gender': gender,
                'insurance_type': insurance,
                'zip_code': zip_code
            })
        
        return pd.DataFrame(patients)

    def generate_admissions(self, patients_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate hospital admissions and readmissions"""
        admissions = []
        readmissions = []
        admission_id = 1
        
        # Start date for data generation
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        for _, patient in patients_df.iterrows():
            # Number of admissions per patient (most have 1-2, some have more)
            n_admissions = np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.25, 0.1, 0.04, 0.01])
            
            patient_admissions = []
            
            for admission_num in range(n_admissions):
                # Admission date
                admission_date = start_date + timedelta(
                    days=np.random.randint(0, (end_date - start_date).days)
                )
                
                # Length of stay (log-normal distribution)
                los = max(1, int(np.random.lognormal(1.5, 0.8)))
                discharge_date = admission_date + timedelta(days=los)
                
                # Risk factors for readmission
                readmission_risk = self._calculate_readmission_risk(
                    patient, admission_num, los
                )
                
                # Determine if readmission occurs
                has_readmission = np.random.random() < readmission_risk
                
                admission_record = {
                    'admission_id': admission_id,
                    'patient_id': patient['patient_id'],
                    'admission_date': admission_date,
                    'discharge_date': discharge_date,
                    'length_of_stay': los,
                    'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], p=[0.7, 0.2, 0.1]),
                    'discharge_disposition': np.random.choice(['Home', 'SNF', 'Rehab', 'Transfer'], p=[0.7, 0.15, 0.1, 0.05])
                }
                
                admissions.append(admission_record)
                patient_admissions.append(admission_record)
                
                # Generate readmission if applicable
                if has_readmission and discharge_date < end_date - timedelta(days=30):
                    readmission_date = discharge_date + timedelta(
                        days=np.random.randint(1, 31)  # Within 30 days
                    )
                    
                    readmissions.append({
                        'readmission_id': len(readmissions) + 1,
                        'original_admission_id': admission_id,
                        'readmission_date': readmission_date,
                        'days_between': (readmission_date - discharge_date).days
                    })
                
                admission_id += 1
        
        return pd.DataFrame(admissions), pd.DataFrame(readmissions)

    def generate_diagnoses(self, admissions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate diagnosis data for each admission"""
        diagnoses = []
        diagnosis_id = 1
        
        for _, admission in admissions_df.iterrows():
            # Number of diagnoses (1-5 per admission)
            n_diagnoses = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.35, 0.2, 0.1, 0.05])
            
            selected_codes = np.random.choice(
                list(self.icd_codes.keys()), 
                size=n_diagnoses, 
                replace=False
            )
            
            for i, icd_code in enumerate(selected_codes):
                diagnoses.append({
                    'diagnosis_id': diagnosis_id,
                    'admission_id': admission['admission_id'],
                    'icd_code': icd_code,
                    'diagnosis_description': self.icd_codes[icd_code],
                    'primary_diagnosis': i == 0  # First diagnosis is primary
                })
                diagnosis_id += 1
        
        return pd.DataFrame(diagnoses)

    def generate_lab_results(self, admissions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate laboratory test results"""
        lab_results = []
        lab_id = 1
        
        lab_tests = {
            'hemoglobin': (12.0, 16.0, 'g/dL'),
            'glucose': (70, 180, 'mg/dL'),
            'creatinine': (0.6, 1.2, 'mg/dL'),
            'sodium': (135, 145, 'mmol/L'),
            'potassium': (3.5, 5.0, 'mmol/L'),
            'bun': (7, 20, 'mg/dL'),
            'wbc_count': (4.0, 11.0, 'K/uL')
        }
        
        for _, admission in admissions_df.iterrows():
            # Generate 3-7 lab tests per admission
            n_tests = np.random.randint(3, 8)
            selected_tests = np.random.choice(
                list(lab_tests.keys()), 
                size=n_tests, 
                replace=False
            )
            
            for test_name in selected_tests:
                min_val, max_val, unit = lab_tests[test_name]
                
                # Some values are abnormal (higher readmission risk)
                if np.random.random() < 0.3:  # 30% abnormal
                    if np.random.random() < 0.5:
                        test_value = np.random.uniform(min_val * 0.5, min_val)  # Low
                    else:
                        test_value = np.random.uniform(max_val, max_val * 1.5)  # High
                else:
                    test_value = np.random.uniform(min_val, max_val)  # Normal
                
                test_date = admission['admission_date'] + timedelta(
                    days=np.random.randint(0, admission['length_of_stay'])
                )
                
                lab_results.append({
                    'lab_id': lab_id,
                    'admission_id': admission['admission_id'],
                    'test_name': test_name,
                    'test_value': round(test_value, 2),
                    'test_unit': unit,
                    'test_date': test_date
                })
                lab_id += 1
        
        return pd.DataFrame(lab_results)

    def generate_medications(self, admissions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate medication data"""
        medications = []
        med_id = 1
        
        for _, admission in admissions_df.iterrows():
            # Number of medications (1-8 per admission)
            n_meds = np.random.randint(1, 9)
            selected_meds = np.random.choice(
                self.medications, 
                size=n_meds, 
                replace=False
            )
            
            for medication in selected_meds:
                medications.append({
                    'medication_id': med_id,
                    'admission_id': admission['admission_id'],
                    'medication_name': medication,
                    'dosage': f"{np.random.randint(5, 500)}mg",
                    'frequency': np.random.choice(['Daily', 'BID', 'TID', 'QID', 'PRN'])
                })
                med_id += 1
        
        return pd.DataFrame(medications)

    def _age_distribution(self) -> List[float]:
        """Create realistic age distribution for hospital admissions"""
        # Higher probability for older ages (common in hospitals)
        ages = range(18, 90)
        probs = []
        for age in ages:
            if age < 30:
                prob = 0.5
            elif age < 50:
                prob = 1.0
            elif age < 70:
                prob = 2.0
            else:
                prob = 3.0
            probs.append(prob)
        
        # Normalize probabilities
        total = sum(probs)
        return [p/total for p in probs]

    def _calculate_readmission_risk(self, patient: Dict, admission_num: int, los: int) -> float:
        """Calculate readmission risk based on patient factors"""
        base_risk = 0.12  # 12% baseline readmission rate
        
        # Age factor
        if patient['age'] > 75:
            base_risk += 0.05
        elif patient['age'] > 65:
            base_risk += 0.03
        
        # Insurance factor
        if patient['insurance_type'] == 'Medicaid':
            base_risk += 0.03
        elif patient['insurance_type'] == 'Uninsured':
            base_risk += 0.05
        
        # Length of stay factor
        if los > 7:
            base_risk += 0.04
        elif los > 3:
            base_risk += 0.02
        
        # Previous admissions factor
        base_risk += admission_num * 0.02
        
        return min(base_risk, 0.4)  # Cap at 40%

    def generate_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic EHR dataset"""
        print("Generating patients...")
        patients_df = self.generate_patients()
        
        print("Generating admissions and readmissions...")
        admissions_df, readmissions_df = self.generate_admissions(patients_df)
        
        print("Generating diagnoses...")
        diagnoses_df = self.generate_diagnoses(admissions_df)
        
        print("Generating lab results...")
        lab_results_df = self.generate_lab_results(admissions_df)
        
        print("Generating medications...")
        medications_df = self.generate_medications(admissions_df)
        
        print(f"Dataset generated successfully!")
        print(f"- {len(patients_df)} patients")
        print(f"- {len(admissions_df)} admissions")
        print(f"- {len(readmissions_df)} readmissions ({len(readmissions_df)/len(admissions_df)*100:.1f}% rate)")
        print(f"- {len(diagnoses_df)} diagnoses")
        print(f"- {len(lab_results_df)} lab results")
        print(f"- {len(medications_df)} medications")
        
        return {
            'patients': patients_df,
            'admissions': admissions_df,
            'readmissions': readmissions_df,
            'diagnoses': diagnoses_df,
            'lab_results': lab_results_df,
            'medications': medications_df
        }

if __name__ == "__main__":
    # Generate synthetic data
    generator = SyntheticEHRGenerator(n_patients=50000)
    datasets = generator.generate_complete_dataset()
    
    # Save to CSV files
    for table_name, df in datasets.items():
        df.to_csv(f"data/raw/{table_name}.csv", index=False)
        print(f"Saved {table_name}.csv")