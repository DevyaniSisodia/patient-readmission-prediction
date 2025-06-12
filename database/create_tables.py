import sqlite3
import os
import pandas as pd
import random

# Ensure the data folder exists
os.makedirs("../data", exist_ok=True)

# Connect to SQLite DB
conn = sqlite3.connect("../data/patient_readmission.db")
cursor = conn.cursor()

# Drop old table if needed (optional during dev/testing)
cursor.execute("DROP TABLE IF EXISTS admission_features")

# Create the table with expanded features
cursor.execute("""
CREATE TABLE IF NOT EXISTS admission_features (
    patient_id INTEGER PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    time_in_hospital INTEGER,
    num_lab_procedures INTEGER,
    num_medications INTEGER,
    number_diagnoses INTEGER,
    num_procedures INTEGER,
    admission_type_id INTEGER,
    discharge_disposition_id INTEGER,
    race TEXT,
    A1Cresult TEXT,
    insulin TEXT,
    change TEXT,
    readmitted INTEGER
)
""")

# Generate synthetic data
races = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
A1C_options = ["None", "Norm", ">7", ">8"]
insulin_options = ["No", "Up", "Down", "Steady"]
change_options = ["No", "Ch"]

sample_data = []
for i in range(1, 1001):
    age = random.randint(20, 90)
    gender = random.choice(["Male", "Female"])
    time_in_hospital = random.randint(1, 14)
    num_lab_procedures = random.randint(10, 80)
    num_medications = random.randint(1, 30)
    number_diagnoses = random.randint(1, 9)
    num_procedures = random.randint(0, 6)
    admission_type_id = random.randint(1, 8)
    discharge_disposition_id = random.randint(1, 30)
    race = random.choice(races)
    A1Cresult = random.choice(A1C_options)
    insulin = random.choice(insulin_options)
    change = random.choice(change_options)

    # Correlated risk logic
    risk_score = (
        (age - 50) * 0.03 +
        (time_in_hospital - 5) * 0.1 +
        (num_lab_procedures - 40) * 0.02 +
        (num_medications - 15) * 0.05 +
        (number_diagnoses - 4) * 0.1 +
        (1 if gender == 'Male' else 0) * 0.1
    )
    prob_readmit = min(max(0.1 + risk_score / 10.0, 0.05), 0.9)
    readmitted = 1 if random.random() < prob_readmit else 0

    sample_data.append((
        i, age, gender, time_in_hospital, num_lab_procedures,
        num_medications, number_diagnoses, num_procedures,
        admission_type_id, discharge_disposition_id,
        race, A1Cresult, insulin, change, readmitted
    ))

# Insert records
cursor.executemany("""
INSERT INTO admission_features (
    patient_id, age, gender, time_in_hospital, num_lab_procedures,
    num_medications, number_diagnoses, num_procedures,
    admission_type_id, discharge_disposition_id,
    race, A1Cresult, insulin, change, readmitted
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", sample_data)

# Commit and close
conn.commit()
conn.close()

print("✅ Inserted 1000 enriched synthetic records into admission_features table.")
