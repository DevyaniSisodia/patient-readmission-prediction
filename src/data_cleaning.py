"""
Data Cleaning Module for Patient Readmission Prediction
Handles data quality issues, missing values, and outliers
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, db_path: str = 'database/patient_readmission.db'):
        self.db_path = db_path
        self.conn = None
        self.cleaning_stats = {}
        
    def connect_db(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"Connected to database: {self.db_path}")
        
    def disconnect_db(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            
    def get_data_quality_summary(self) -> pd.DataFrame:
        """Get data quality summary from the database view"""
        query = "SELECT * FROM data_quality_summary"
        return pd.read_sql_query(query, self.conn)
    
    def clean_patient_data(self) -> Dict[str, int]:
        """Clean patient demographic data"""
        logger.info("Cleaning patient data...")
        
        stats = {'patients_processed': 0, 'age_outliers_fixed': 0, 'missing_values_fixed': 0}
        
        # Get patients with data quality issues
        query = """
        SELECT patient_id, age, gender, insurance_type, zip_code
        FROM patients 
        WHERE age < 0 OR age > 120 OR age IS NULL
           OR gender IS NULL OR gender NOT IN ('M', 'F')
           OR insurance_type IS NULL
        """
        
        problematic_patients = pd.read_sql_query(query, self.conn)
        stats['patients_processed'] = len(problematic_patients)
        
        if len(problematic_patients) > 0:
            # Fix age outliers
            age_mask = (problematic_patients['age'] < 0) | (problematic_patients['age'] > 120)
            if age_mask.any():
                # Replace with median age from valid patients
                median_age_query = "SELECT AVG(age) as median_age FROM patients WHERE age BETWEEN 18 AND 100"
                median_age = pd.read_sql_query(median_age_query, self.conn)['median_age'].iloc[0]
                
                for patient_id in problematic_patients[age_mask]['patient_id']:
                    self.conn.execute(
                        "UPDATE patients SET age = ? WHERE patient_id = ?",
                        (int(median_age), patient_id)
                    )
                    stats['age_outliers_fixed'] += 1
            
            # Fix missing gender (assign randomly based on distribution)
            gender_null_mask = problematic_patients['gender'].isnull()
            if gender_null_mask.any():
                for patient_id in problematic_patients[gender_null_mask]['patient_id']:
                    random_gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
                    self.conn.execute(
                        "UPDATE patients SET gender = ? WHERE patient_id = ?",
                        (random_gender, patient_id)
                    )
                    stats['missing_values_fixed'] += 1
            
            # Fix missing insurance type
            insurance_null_mask = problematic_patients['insurance_type'].isnull()
            if insurance_null_mask.any():
                for patient_id in problematic_patients[insurance_null_mask]['patient_id']:
                    random_insurance = np.random.choice(
                        ['Medicare', 'Medicaid', 'Private', 'Uninsured'], 
                        p=[0.4, 0.2, 0.35, 0.05]
                    )
                    self.conn.execute(
                        "UPDATE patients SET insurance_type = ? WHERE patient_id = ?",
                        (random_insurance, patient_id)
                    )
                    stats['missing_values_fixed'] += 1
        
        self.conn.commit()
        logger.info(f"Patient data cleaning completed: {stats}")
        return stats
    
    def clean_admission_data(self) -> Dict[str, int]:
        """Clean admission data for inconsistencies"""
        logger.info("Cleaning admission data...")
        
        stats = {'admissions_processed': 0, 'date_issues_fixed': 0, 'los_issues_fixed': 0}
        
        # Find admissions with date/LOS issues
        query = """
        SELECT admission_id, patient_id, admission_date, discharge_date, length_of_stay
        FROM admissions 
        WHERE discharge_date < admission_date 
           OR length_of_stay <= 0
           OR length_of_stay != (julianday(discharge_date) - julianday(admission_date))
        """
        
        problematic_admissions = pd.read_sql_query(query, self.conn)
        stats['admissions_processed'] = len(problematic_admissions)
        
        for _, admission in problematic_admissions.iterrows():
            admission_id = admission['admission_id']
            admission_date = pd.to_datetime(admission['admission_date'])
            discharge_date = pd.to_datetime(admission['discharge_date'])
            
            # Fix date order issues
            if discharge_date < admission_date:
                # Swap dates if discharge is before admission
                self.conn.execute("""
                    UPDATE admissions 
                    SET admission_date = ?, discharge_date = ?
                    WHERE admission_id = ?
                """, (discharge_date.strftime('%Y-%m-%d'), 
                      admission_date.strftime('%Y-%m-%d'), admission_id))
                stats['date_issues_fixed'] += 1
            
            # Recalculate length of stay
            corrected_los = (pd.to_datetime(discharge_date) - pd.to_datetime(admission_date)).days
            if corrected_los <= 0:
                corrected_los = 1  # Minimum 1 day stay
                
            self.conn.execute("""
                UPDATE admissions 
                SET length_of_stay = ?
                WHERE admission_id = ?
            """, (corrected_los, admission_id))
            stats['los_issues_fixed'] += 1
        
        self.conn.commit()
        logger.info(f"Admission data cleaning completed: {stats}")
        return stats
    
    def clean_lab_data(self) -> Dict[str, int]:
        """Clean laboratory results data"""
        logger.info("Cleaning lab data...")
        
        stats = {'labs_processed': 0, 'outliers_removed': 0, 'abnormal_flags_fixed': 0}
        
        # Define reasonable ranges for common lab tests
        lab_ranges = {
            'Hemoglobin': (4.0, 20.0),
            'Creatinine': (0.1, 15.0),
            'BUN': (2, 150),
            'Glucose': (20, 800),
            'Sodium': (120, 160),
            'Potassium': (1.5, 8.0),
            'Chloride': (80, 120),
            'CO2': (10, 40)
        }
        
        for test_name, (min_val, max_val) in lab_ranges.items():
            # Find outliers
            query = f"""
            SELECT lab_id, test_value, reference_range_low, reference_range_high
            FROM lab_results 
            WHERE test_name = '{test_name}' 
            AND (test_value < {min_val} OR test_value > {max_val})
            """
            
            outliers = pd.read_sql_query(query, self.conn)
            stats['labs_processed'] += len(outliers)
            
            # Remove extreme outliers
            for lab_id in outliers['lab_id']:
                self.conn.execute("DELETE FROM lab_results WHERE lab_id = ?", (lab_id,))
                stats['outliers_removed'] += 1
        
        # Fix abnormal flags based on reference ranges
        query = """
        SELECT lab_id, test_value, reference_range_low, reference_range_high, abnormal_flag
        FROM lab_results 
        WHERE reference_range_low IS NOT NULL 
        AND reference_range_high IS NOT NULL
        AND abnormal_flag IS NOT NULL
        """
        
        labs_to_check = pd.read_sql_query(query, self.conn)
        
        for _, lab in labs_to_check.iterrows():
            correct_flag = 'N'  # Normal
            if lab['test_value'] < lab['reference_range_low']:
                correct_flag = 'L'  # Low
            elif lab['test_value'] > lab['reference_range_high']:
                correct_flag = 'H'  # High
                
            if correct_flag != lab['abnormal_flag']:
                self.conn.execute("""
                    UPDATE lab_results 
                    SET abnormal_flag = ?
                    WHERE lab_id = ?
                """, (correct_flag, lab['lab_id']))
                stats['abnormal_flags_fixed'] += 1
        
        self.conn.commit()
        logger.info(f"Lab data cleaning completed: {stats}")
        return stats
    
    def remove_duplicate_records(self) -> Dict[str, int]:
        """Remove duplicate records across all tables"""
        logger.info("Removing duplicate records...")
        
        stats = {}
        
        # Remove duplicate diagnoses
        duplicate_diagnoses_query = """
        DELETE FROM diagnoses 
        WHERE diagnosis_id NOT IN (
            SELECT MIN(diagnosis_id)
            FROM diagnoses 
            GROUP BY admission_id, icd_code
        )
        """
        result = self.conn.execute(duplicate_diagnoses_query)
        stats['duplicate_diagnoses_removed'] = result.rowcount
        
        # Remove duplicate medications
        duplicate_meds_query = """
        DELETE FROM medications 
        WHERE medication_id NOT IN (
            SELECT MIN(medication_id)
            FROM medications 
            GROUP BY admission_id, medication_name, dosage
        )
        """
        result = self.conn.execute(duplicate_meds_query)
        stats['duplicate_medications_removed'] = result.rowcount
        
        # Remove duplicate procedures
        duplicate_procedures_query = """
        DELETE FROM procedures 
        WHERE procedure_id NOT IN (
            SELECT MIN(procedure_id)
            FROM procedures 
            GROUP BY admission_id, procedure_code, procedure_date
        )
        """
        result = self.conn.execute(duplicate_procedures_query)
        stats['duplicate_procedures_removed'] = result.rowcount
        
        self.conn.commit()
        logger.info(f"Duplicate removal completed: {stats}")
        return stats
    
    def validate_referential_integrity(self) -> Dict[str, int]:
        """Check and fix referential integrity issues"""
        logger.info("Validating referential integrity...")
        
        stats = {}
        
        # Remove orphaned admissions (patients that don't exist)
        orphan_admissions_query = """
        DELETE FROM admissions 
        WHERE patient_id NOT IN (SELECT patient_id FROM patients)
        """
        result = self.conn.execute(orphan_admissions_query)
        stats['orphaned_admissions_removed'] = result.rowcount
        
        # Remove orphaned diagnoses
        orphan_diagnoses_query = """
        DELETE FROM diagnoses 
        WHERE admission_id NOT IN (SELECT admission_id FROM admissions)
        """
        result = self.conn.execute(orphan_diagnoses_query)
        stats['orphaned_diagnoses_removed'] = result.rowcount
        
        # Remove orphaned lab results
        orphan_labs_query = """
        DELETE FROM lab_results 
        WHERE admission_id NOT IN (SELECT admission_id FROM admissions)
        """
        result = self.conn.execute(orphan_labs_query)
        stats['orphaned_labs_removed'] = result.rowcount
        
        # Remove orphaned medications
        orphan_meds_query = """
        DELETE FROM medications 
        WHERE admission_id NOT IN (SELECT admission_id FROM admissions)
        """
        result = self.conn.execute(orphan_meds_query)
        stats['orphaned_medications_removed'] = result.rowcount
        
        # Remove orphaned procedures
        orphan_procedures_query = """
        DELETE FROM procedures 
        WHERE admission_id NOT IN (SELECT admission_id FROM admissions)
        """
        result = self.conn.execute(orphan_procedures_query)
        stats['orphaned_procedures_removed'] = result.rowcount
        
        # Remove orphaned readmissions
        orphan_readmissions_query = """
        DELETE FROM readmissions 
        WHERE original_admission_id NOT IN (SELECT admission_id FROM admissions)
        """
        result = self.conn.execute(orphan_readmissions_query)
        stats['orphaned_readmissions_removed'] = result.rowcount
        
        self.conn.commit()
        logger.info(f"Referential integrity validation completed: {stats}")
        return stats
    
    def get_cleaning_summary(self) -> pd.DataFrame:
        """Get summary of all cleaning operations"""
        if not self.cleaning_stats:
            return pd.DataFrame()
            
        summary_data = []
        for operation, stats in self.cleaning_stats.items():
            for metric, value in stats.items():
                summary_data.append({
                    'Operation': operation,
                    'Metric': metric,
                    'Count': value
                })
        
        return pd.DataFrame(summary_data)
    
    def run_full_cleaning_pipeline(self) -> Dict[str, Dict]:
        """Run the complete data cleaning pipeline"""
        logger.info("Starting full data cleaning pipeline...")
        
        self.connect_db()
        
        try:
            # Store all cleaning statistics
            self.cleaning_stats['patient_cleaning'] = self.clean_patient_data()
            self.cleaning_stats['admission_cleaning'] = self.clean_admission_data()
            self.cleaning_stats['lab_cleaning'] = self.clean_lab_data()
            self.cleaning_stats['duplicate_removal'] = self.remove_duplicate_records()
            self.cleaning_stats['integrity_validation'] = self.validate_referential_integrity()
            
            logger.info("Data cleaning pipeline completed successfully!")
            return self.cleaning_stats
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {e}")
            raise
        finally:
            self.disconnect_db()

def main():
    """Main function to run data cleaning"""
    cleaner = DataCleaner()
    
    try:
        # Run full cleaning pipeline
        results = cleaner.run_full_cleaning_pipeline()
        
        # Print summary
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        
        for operation, stats in results.items():
            print(f"\n{operation.upper().replace('_', ' ')}:")
            for metric, value in stats.items():
                print(f"  {metric.replace('_', ' ').title()}: {value}")
        
        print("\n" + "="*50)
        print("Data cleaning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()