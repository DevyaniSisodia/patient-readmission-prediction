-- Patient Readmission Risk Prediction Database Schema
-- Optimized for analytical queries and machine learning feature extraction

-- Drop existing tables if they exist
DROP TABLE IF EXISTS medications;
DROP TABLE IF EXISTS lab_results;
DROP TABLE IF EXISTS procedures;
DROP TABLE IF EXISTS diagnoses;
DROP TABLE IF EXISTS readmissions;
DROP TABLE IF EXISTS admissions;
DROP TABLE IF EXISTS patients;

-- Patients table - Core demographic information
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    age INTEGER NOT NULL CHECK(age >= 0 AND age <= 120),
    gender TEXT NOT NULL CHECK(gender IN ('M', 'F')),
    insurance_type TEXT NOT NULL CHECK(insurance_type IN ('Medicare', 'Medicaid', 'Private', 'Uninsured')),
    zip_code TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Admissions table - Hospital stay information
CREATE TABLE admissions (
    admission_id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL,
    admission_date DATE NOT NULL,
    discharge_date DATE NOT NULL,
    length_of_stay INTEGER NOT NULL CHECK(length_of_stay > 0),
    admission_type TEXT NOT NULL CHECK(admission_type IN ('Emergency', 'Elective', 'Urgent')),
    discharge_disposition TEXT NOT NULL CHECK(discharge_disposition IN ('Home', 'SNF', 'Rehab', 'Transfer', 'Deceased')),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    CHECK(discharge_date >= admission_date)
);

-- Diagnoses table - Medical conditions per admission
CREATE TABLE diagnoses (
    diagnosis_id INTEGER PRIMARY KEY,
    admission_id INTEGER NOT NULL,
    icd_code TEXT NOT NULL,
    diagnosis_description TEXT NOT NULL,
    primary_diagnosis BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
);

-- Lab Results table - Laboratory test values
CREATE TABLE lab_results (
    lab_id INTEGER PRIMARY KEY,
    admission_id INTEGER NOT NULL,
    test_name TEXT NOT NULL,
    test_value REAL NOT NULL,
    test_unit TEXT NOT NULL,
    test_date DATE NOT NULL,
    reference_range_low REAL,
    reference_range_high REAL,
    abnormal_flag TEXT CHECK(abnormal_flag IN ('L', 'H', 'N')),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
);

-- Medications table - Prescribed medications
CREATE TABLE medications (
    medication_id INTEGER PRIMARY KEY,
    admission_id INTEGER NOT NULL,
    medication_name TEXT NOT NULL,
    dosage TEXT NOT NULL,
    frequency TEXT NOT NULL CHECK(frequency IN ('Daily', 'BID', 'TID', 'QID', 'PRN')),
    start_date DATE,
    end_date DATE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
);

-- Procedures table - Medical procedures performed
CREATE TABLE procedures (
    procedure_id INTEGER PRIMARY KEY,
    admission_id INTEGER NOT NULL,
    procedure_code TEXT NOT NULL,
    procedure_description TEXT NOT NULL,
    procedure_date DATE NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
);

-- Readmissions table - 30-day readmission tracking
CREATE TABLE readmissions (
    readmission_id INTEGER PRIMARY KEY,
    original_admission_id INTEGER NOT NULL,
    readmission_date DATE NOT NULL,
    days_between INTEGER NOT NULL CHECK(days_between > 0 AND days_between <= 30),
    readmission_admission_id INTEGER,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (original_admission_id) REFERENCES admissions(admission_id),
    FOREIGN KEY (readmission_admission_id) REFERENCES admissions(admission_id)
);

-- Create indexes for query optimization
-- Primary lookup indexes
CREATE INDEX idx_patient_admissions ON admissions(patient_id, admission_date);
CREATE INDEX idx_admission_diagnoses ON diagnoses(admission_id);
CREATE INDEX idx_admission_labs ON lab_results(admission_id, test_date);
CREATE INDEX idx_admission_medications ON medications(admission_id);
CREATE INDEX idx_admission_procedures ON procedures(admission_id);
CREATE INDEX idx_readmission_lookup ON readmissions(original_admission_id);

-- Date-based indexes for temporal queries
CREATE INDEX idx_admission_date ON admissions(admission_date);
CREATE INDEX idx_discharge_date ON admissions(discharge_date);
CREATE INDEX idx_lab_test_date ON lab_results(test_date);

-- Diagnostic indexes for analysis
CREATE INDEX idx_primary_diagnosis ON diagnoses(icd_code) WHERE primary_diagnosis = TRUE;
CREATE INDEX idx_lab_test_name ON lab_results(test_name);
CREATE INDEX idx_medication_name ON medications(medication_name);

-- Composite indexes for complex queries
CREATE INDEX idx_patient_age_insurance ON patients(age, insurance_type);
CREATE INDEX idx_admission_type_los ON admissions(admission_type, length_of_stay);

-- Views for common analytical queries

-- Patient summary view with key metrics
CREATE VIEW patient_summary AS
SELECT 
    p.patient_id,
    p.age,
    p.gender,
    p.insurance_type,
    COUNT(a.admission_id) as total_admissions,
    COUNT(r.readmission_id) as total_readmissions,
    ROUND(COUNT(r.readmission_id) * 100.0 / COUNT(a.admission_id), 2) as readmission_rate,
    AVG(a.length_of_stay) as avg_length_of_stay,
    MIN(a.admission_date) as first_admission,
    MAX(a.discharge_date) as last_discharge
FROM patients p
LEFT JOIN admissions a ON p.patient_id = a.patient_id
LEFT JOIN readmissions r ON a.admission_id = r.original_admission_id
GROUP BY p.patient_id, p.age, p.gender, p.insurance_type;

-- Admission features view for ML training
CREATE VIEW admission_features AS
SELECT 
    a.admission_id,
    a.patient_id,
    p.age,
    p.gender,
    p.insurance_type,
    a.admission_date,
    a.discharge_date,
    a.length_of_stay,
    a.admission_type,
    a.discharge_disposition,
    
    -- Previous admission history
    COUNT(prev_a.admission_id) as previous_admissions,
    AVG(prev_a.length_of_stay) as avg_previous_los,
    
    -- Diagnosis complexity
    COUNT(d.diagnosis_id) as diagnosis_count,
    COUNT(CASE WHEN d.primary_diagnosis = TRUE THEN 1 END) as primary_diagnosis_count,
    
    -- Lab abnormalities
    COUNT(l.lab_id) as lab_test_count,
    COUNT(CASE WHEN l.abnormal_flag IN ('L', 'H') THEN 1 END) as abnormal_lab_count,
    
    -- Medication complexity
    COUNT(m.medication_id) as medication_count,
    
    -- Procedure complexity
    COUNT(pr.procedure_id) as procedure_count,
    
    -- Target variable: 30-day readmission
    CASE WHEN r.readmission_id IS NOT NULL THEN 1 ELSE 0 END as readmission_30_day
    
FROM admissions a
JOIN patients p ON a.patient_id = p.patient_id
LEFT JOIN admissions prev_a ON a.patient_id = prev_a.patient_id 
    AND prev_a.discharge_date < a.admission_date
LEFT JOIN diagnoses d ON a.admission_id = d.admission_id
LEFT JOIN lab_results l ON a.admission_id = l.admission_id
LEFT JOIN medications m ON a.admission_id = m.admission_id
LEFT JOIN procedures pr ON a.admission_id = pr.admission_id
LEFT JOIN readmissions r ON a.admission_id = r.original_admission_id
GROUP BY a.admission_id, a.patient_id, p.age, p.gender, p.insurance_type,
         a.admission_date, a.discharge_date, a.length_of_stay, 
         a.admission_type, a.discharge_disposition, r.readmission_id;

-- High-risk diagnosis codes view
CREATE VIEW high_risk_diagnoses AS
SELECT 
    icd_code,
    diagnosis_description,
    COUNT(d.diagnosis_id) as diagnosis_count,
    COUNT(r.readmission_id) as readmission_count,
    ROUND(COUNT(r.readmission_id) * 100.0 / COUNT(d.diagnosis_id), 2) as readmission_rate
FROM diagnoses d
JOIN admissions a ON d.admission_id = a.admission_id
LEFT JOIN readmissions r ON a.admission_id = r.original_admission_id
GROUP BY icd_code, diagnosis_description
HAVING COUNT(d.diagnosis_id) >= 10  -- Only codes with sufficient frequency
ORDER BY readmission_rate DESC;

-- Lab values summary for risk assessment
CREATE VIEW lab_risk_summary AS
SELECT 
    test_name,
    COUNT(l.lab_id) as test_count,
    AVG(l.test_value) as avg_value,
    MIN(l.test_value) as min_value,
    MAX(l.test_value) as max_value,
    COUNT(CASE WHEN l.abnormal_flag IN ('L', 'H') THEN 1 END) as abnormal_count,
    COUNT(r.readmission_id) as readmission_count,
    ROUND(COUNT(r.readmission_id) * 100.0 / COUNT(DISTINCT a.admission_id), 2) as readmission_rate
FROM lab_results l
JOIN admissions a ON l.admission_id = a.admission_id
LEFT JOIN readmissions r ON a.admission_id = r.original_admission_id
GROUP BY test_name
ORDER BY readmission_rate DESC;

-- Medication risk patterns
CREATE VIEW medication_risk_patterns AS
SELECT 
    medication_name,
    COUNT(m.medication_id) as prescription_count,
    COUNT(DISTINCT m.admission_id) as unique_admissions,
    COUNT(r.readmission_id) as readmission_count,
    ROUND(COUNT(r.readmission_id) * 100.0 / COUNT(DISTINCT m.admission_id), 2) as readmission_rate
FROM medications m
JOIN admissions a ON m.admission_id = a.admission_id
LEFT JOIN readmissions r ON a.admission_id = r.original_admission_id
GROUP BY medication_name
HAVING COUNT(DISTINCT m.admission_id) >= 10
ORDER BY readmission_rate DESC;

-- Data quality checks
CREATE VIEW data_quality_summary AS
SELECT 
    'patients' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN age IS NULL THEN 1 END) as null_age,
    COUNT(CASE WHEN gender IS NULL THEN 1 END) as null_gender,
    COUNT(CASE WHEN insurance_type IS NULL THEN 1 END) as null_insurance,
    0 as null_dates
FROM patients
UNION ALL
SELECT 
    'admissions' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN length_of_stay IS NULL THEN 1 END) as null_los,
    0 as null_gender,
    0 as null_insurance,
    COUNT(CASE WHEN admission_date IS NULL OR discharge_date IS NULL THEN 1 END) as null_dates
FROM admissions
UNION ALL
SELECT 
    'diagnoses' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN icd_code IS NULL THEN 1 END) as null_icd,
    0 as null_gender,
    0 as null_insurance,
    0 as null_dates
FROM diagnoses;

-- Performance monitoring queries
-- Query to check index usage
-- EXPLAIN QUERY PLAN SELECT * FROM admission_features WHERE patient_id = 12345;

-- Query to check readmission distribution
-- SELECT readmission_30_day, COUNT(*) FROM admission_features GROUP BY readmission_30_day;