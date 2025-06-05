"""
Feature Engineering Pipeline for Patient Readmission Prediction
Transforms raw EHR data into ML-ready features with optimized SQL queries
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, db_path: str = "database/patient_readmission.db"):
        """
        Initialize Feature Engineering Pipeline
        
        Args:
            db_path: Path to SQLite database
        """
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.feature_queries = self._define_feature_queries()
    
    def _define_feature_queries(self) -> Dict[str, str]:
        """Define optimized SQL queries for feature extraction"""
        
        queries = {
            'base_features': """
            SELECT 
                a.admission_id,
                a.patient_id,
                p.age,
                CASE 
                    WHEN p.age < 30 THEN 'Young'
                    WHEN p.age < 50 THEN 'Middle'
                    WHEN p.age < 70 THEN 'Senior'
                    ELSE 'Elderly'
                END as age_group,
                p.gender,
                p.insurance_type,
                a.admission_date,
                a.discharge_date,
                a.length_of_stay,
                CASE WHEN a.length_of_stay > 7 THEN 1 ELSE 0 END as long_stay,
                a.admission_type,
                a.discharge_disposition,
                -- Day of week features (weekends vs weekdays)
                CASE WHEN strftime('%w', a.admission_date) IN ('0', '6') THEN 1 ELSE 0 END as weekend_admission,
                CASE WHEN strftime('%w', a.discharge_date) IN ('0', '6') THEN 1 ELSE 0 END as weekend_discharge,
                -- Season of admission
                CASE 
                    WHEN CAST(strftime('%m', a.admission_date) AS INTEGER) IN (12, 1, 2) THEN 'Winter'
                    WHEN CAST(strftime('%m', a.admission_date) AS INTEGER) IN (3, 4, 5) THEN 'Spring'
                    WHEN CAST(strftime('%m', a.admission_date) AS INTEGER) IN (6, 7, 8) THEN 'Summer'
                    ELSE 'Fall'
                END as admission_season
            FROM admissions a
            JOIN patients p ON a.patient_id = p.patient_id
            """,
            
            'admission_history': """
            SELECT 
                curr.admission_id,
                COUNT(prev.admission_id) as previous_admissions,
                COALESCE(AVG(prev.length_of_stay), 0) as avg_previous_los,
                COALESCE(MAX(prev.length_of_stay), 0) as max_previous_los,
                COALESCE(
                    ROUND(
                        julianday(curr.admission_date) - 
                        MAX(julianday(prev.discharge_date))
                    ), 365
                ) as days_since_last_discharge,
                -- Readmission history
                COUNT(r.readmission_id) as previous_readmissions,
                CASE WHEN COUNT(r.readmission_id) > 0 THEN 1 ELSE 0 END as has_readmission_history
            FROM admissions curr
            LEFT JOIN admissions prev ON curr.patient_id = prev.patient_id 
                AND prev.discharge_date < curr.admission_date
            LEFT JOIN readmissions r ON prev.admission_id = r.original_admission_id
            GROUP BY curr.admission_id
            """,
            
            'diagnosis_features': """
            SELECT 
                a.admission_id,
                COUNT(d.diagnosis_id) as diagnosis_count,
                COUNT(CASE WHEN d.primary_diagnosis = 1 THEN 1 END) as primary_diagnosis_count,
                -- High-risk conditions
                MAX(CASE WHEN d.icd_code LIKE 'I50%' THEN 1 ELSE 0 END) as heart_failure,
                MAX(CASE WHEN d.icd_code LIKE 'E11%' THEN 1 ELSE 0 END) as diabetes,
                MAX(CASE WHEN d.icd_code LIKE 'I10%' THEN 1 ELSE 0 END) as hypertension,
                MAX(CASE WHEN d.icd_code LIKE 'J44%' THEN 1 ELSE 0 END) as copd,
                MAX(CASE WHEN d.icd_code LIKE 'N18%' THEN 1 ELSE 0 END) as kidney_disease,
                MAX(CASE WHEN d.icd_code LIKE 'F03%' THEN 1 ELSE 0 END) as dementia,
                MAX(CASE WHEN d.icd_code LIKE 'I25%' THEN 1 ELSE 0 END) as coronary_disease,
                -- Comorbidity score (simplified Charlson)
                (MAX(CASE WHEN d.icd_code LIKE 'I50%' THEN 1 ELSE 0 END) +
                 MAX(CASE WHEN d.icd_code LIKE 'E11%' THEN 1 ELSE 0 END) +
                 MAX(CASE WHEN d.icd_code LIKE 'I10%' THEN 1 ELSE 0 END) +
                 MAX(CASE WHEN d.icd_code LIKE 'J44%' THEN 1 ELSE 0 END) +
                 MAX(CASE WHEN d.icd_code LIKE 'N18%' THEN 1 ELSE 0 END) +
                 MAX(CASE WHEN d.icd_code LIKE 'F03%' THEN 1 ELSE 0 END)) as comorbidity_score
            FROM admissions a
            LEFT JOIN diagnoses d ON a.admission_id = d.admission_id
            GROUP BY a.admission_id
            """,
            
            'lab_features': """
            SELECT 
                a.admission_id,
                COUNT(l.lab_id) as lab_test_count,
                COUNT(CASE WHEN l.abnormal_flag IN ('L', 'H') THEN 1 END) as abnormal_lab_count,
                CASE WHEN COUNT(l.lab_id) > 0 THEN 
                    ROUND(COUNT(CASE WHEN l.abnormal_flag IN ('L', 'H') THEN 1 END) * 100.0 / COUNT(l.lab_id), 2)
                ELSE 0 END as abnormal_lab_percentage,
                
                -- Key lab values (using latest value per admission)
                MAX(CASE WHEN l.test_name = 'hemoglobin' THEN l.test_value END) as hemoglobin,
                MAX(CASE WHEN l.test_name = 'glucose' THEN l.test_value END) as glucose,
                MAX(CASE WHEN l.test_name = 'creatinine' THEN l.test_value END) as creatinine,
                MAX(CASE WHEN l.test_name = 'sodium' THEN l.test_value END) as sodium,
                MAX(CASE WHEN l.test_name = 'potassium' THEN l.test_value END) as potassium,
                MAX(CASE WHEN l.test_name = 'bun' THEN l.test_value END) as bun,
                MAX(CASE WHEN l.test_name = 'wbc_count' THEN l.test_value END) as wbc_count,
                
                -- Lab abnormality flags
                MAX(CASE WHEN l.test_name = 'hemoglobin' AND l.test_value < 12 THEN 1 ELSE 0 END) as low_hemoglobin,
                MAX(CASE WHEN l.test_name = 'glucose' AND l.test_value > 180 THEN 1 ELSE 0 END) as high_glucose,
                MAX(CASE WHEN l.test_name = 'creatinine' AND l.test_value > 1.2 THEN 1 ELSE 0 END) as high_creatinine,
                MAX(CASE WHEN l.test_name = 'wbc_count' AND (l.test_value < 4 OR l.test_value > 11) THEN 1 ELSE 0 END) as abnormal_wbc
            FROM admissions a
            LEFT JOIN lab_results l ON a.admission_id = l.admission_id
            GROUP BY a.admission_id
            """,
            
            'medication_features': """
            SELECT 
                a.admission_id,
                COUNT(m.medication_id) as medication_count,
                COUNT(DISTINCT m.medication_name) as unique_medications,
                -- High-risk medications
                MAX(CASE WHEN m.medication_name IN ('Warfarin', 'Insulin') THEN 1 ELSE 0 END) as high_risk_meds,
                MAX(CASE WHEN m.medication_name LIKE '%insulin%' THEN 1 ELSE 0 END) as on_insulin,
                MAX(CASE WHEN m.medication_name IN ('Furosemide') THEN 1 ELSE 0 END) as on_diuretics,
                MAX(CASE WHEN m.medication_name IN ('Prednisone') THEN 1 ELSE 0 END) as on_steroids,
                -- Polypharmacy (5+ medications is high risk)
                CASE WHEN COUNT(DISTINCT m.medication_name) >= 5 THEN 1 ELSE 0 END as polypharmacy
            FROM admissions a
            LEFT JOIN medications m ON a.admission_id = m.admission_id
            GROUP BY a.admission_id
            """,
            
            'readmission_target': """
            SELECT 
                a.admission_id,
                CASE WHEN r.readmission_id IS NOT NULL THEN 1 ELSE 0 END as readmission_30_day,
                r.days_between as days_to_readmission
            FROM admissions a
            LEFT JOIN readmissions r ON a.admission_id = r.original_admission_id
            """
        }
        
        return queries
    
    def extract_features(self) -> pd.DataFrame:
        """
        Execute all feature extraction queries and combine results
        Returns: Complete feature dataset ready for ML
        """
        print("Extracting features from database...")
        
        # Execute each feature query
        feature_dfs = {}
        for feature_name, query in self.feature_queries.items():
            print(f"  - Extracting {feature_name}...")
            feature_dfs[feature_name] = pd.read_sql(query, self.engine)
        
        # Combine all features on admission_id
        print("Combining feature sets...")
        combined_df = feature_dfs['base_features']
        
        for feature_name, df in feature_dfs.items():
            if feature_name != 'base_features':
                combined_df = combined_df.merge(df, on='admission_id', how='left')
        
        print(f"Feature extraction complete: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        return combined_df
    
    def engineer_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features using pandas operations
        
        Args:
            df: Base feature dataframe
        Returns: Enhanced feature dataframe
        """
        print("Engineering additional features...")
        
        # Convert dates to datetime
        df['admission_date'] = pd.to_datetime(df['admission_date'])
        df['discharge_date'] = pd.to_datetime(df['discharge_date'])
        
        # Time-based features
        df['admission_hour'] = df['admission_date'].dt.hour
        df['admission_month'] = df['admission_date'].dt.month
        df['admission_quarter'] = df['admission_date'].dt.quarter
        df['admission_day_of_year'] = df['admission_date'].dt.dayofyear
        
        # Risk scores
        df['age_risk_score'] = np.where(df['age'] > 75, 3,
                                      np.where(df['age'] > 65, 2,
                                             np.where(df['age'] > 50, 1, 0)))
        
        df['comorbidity_risk'] = (df['comorbidity_score'] * 2 + 
                                df['diagnosis_count'] * 0.5 +
                                df['abnormal_lab_count'] * 0.3)
        
        # Interaction features
        df['age_los_interaction'] = df['age'] * df['length_of_stay']
        df['comorbidity_los_interaction'] = df['comorbidity_score'] * df['length_of_stay']
        df['previous_admissions_age'] = df['previous_admissions'] * df['age']
        
        # Categorical combinations
        df['age_insurance_combo'] = df['age_group'] + '_' + df['insurance_type']
        df['admission_discharge_combo'] = df['admission_type'] + '_' + df['discharge_disposition']
        
        # Fill missing values with appropriate defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['admission_date', 'discharge_date']:
                df[col] = df[col].fillna('Unknown')
        
        print(f"Additional feature engineering complete: {df.shape[1]} total features")
        return df
    
    def create_ml_dataset(self, test_size: float = 0.2, 
                         validation_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits with temporal validation
        
        Args:
            test_size: Proportion for test set
            validation_size: Proportion for validation set
        Returns: (train_df, val_df, test_df)
        """
        print("Creating ML dataset with temporal splits...")
        
        # Extract and engineer features
        df = self.extract_features()
        df = self.engineer_additional_features(df)
        
        # Sort by admission date for temporal splitting
        df = df.sort_values('admission_date')
        
        # Calculate split indices
        n_total = len(df)
        n_test = int(n_total * test_size)
        n_val = int(n_total * validation_size)
        n_train = n_total - n_test - n_val
        
        # Create temporal splits (oldest for training, newest for testing)
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train+n_val].copy()
        test_df = df.iloc[n_train+n_val:].copy()
        
        print(f"Dataset splits:")
        print(f"  - Training: {len(train_df)} samples ({train_df['readmission_30_day'].mean():.3f} readmission rate)")
        print(f"  - Validation: {len(val_df)} samples ({val_df['readmission_30_day'].mean():.3f} readmission rate)")
        print(f"  - Test: {len(test_df)} samples ({test_df['readmission_30_day'].mean():.3f} readmission rate)")
        
        return train_df, val_df, test_df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by category for interpretability analysis
        Returns: Dictionary mapping feature groups to feature names
        """
        feature_groups = {
            'demographics': ['age', 'gender', 'insurance_type', 'age_group', 'age_risk_score'],
            'admission_characteristics': ['length_of_stay', 'admission_type', 'discharge_disposition', 
                                        'long_stay', 'weekend_admission', 'weekend_discharge'],
            'medical_history': ['previous_admissions', 'avg_previous_los', 'max_previous_los',
                              'days_since_last_discharge', 'previous_readmissions', 'has_readmission_history'],
            'diagnoses': ['diagnosis_count', 'heart_failure', 'diabetes', 'hypertension', 'copd',
                         'kidney_disease', 'dementia', 'coronary_disease', 'comorbidity_score'],
            'lab_results': ['lab_test_count', 'abnormal_lab_count', 'abnormal_lab_percentage',
                          'hemoglobin', 'glucose', 'creatinine', 'low_hemoglobin', 'high_glucose'],
            'medications': ['medication_count', 'unique_medications', 'high_risk_meds', 
                          'on_insulin', 'polypharmacy'],
            'temporal': ['admission_season', 'admission_month', 'admission_quarter'],
            'risk_scores': ['comorbidity_risk', 'age_los_interaction', 'comorbidity_los_interaction']
        }
        return feature_groups
    
    def generate_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for all features
        
        Args:
            df: Feature dataframe
        Returns: Feature summary dataframe
        """
        summary_data = []
        
        for col in df.columns:
            if col in ['admission_id', 'patient_id', 'admission_date', 'discharge_date']:
                continue
                
            col_info = {
                'feature_name': col,
                'data_type': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': df[col].isnull().mean() * 100,
                'unique_values': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q50': df[col].quantile(0.50),
                    'q75': df[col].quantile(0.75)
                })
            else:
                col_info.update({
                    'mean': None, 'std': None, 'min': None, 'max': None,
                    'q25': None, 'q50': None, 'q75': None
                })
            
            summary_data.append(col_info)
        
        return pd.DataFrame(summary_data)

if __name__ == "__main__":
    # Example usage
    fe = FeatureEngineer()
    
    # Create ML-ready dataset
    train_df, val_df, test_df = fe.create_ml_dataset()
    
    # Save datasets
    train_df.to_csv('data/processed/train_features.csv', index=False)
    val_df.to_csv('data/processed/val_features.csv', index=False)
    test_df.to_csv('data/processed/test_features.csv', index=False)
    
    # Generate feature summary
    feature_summary = fe.generate_feature_summary(train_df)
    feature_summary.to_csv('data/processed/feature_summary.csv', index=False)
    
    print("Feature engineering pipeline completed successfully!")
    print(f"Files saved to data/processed/")