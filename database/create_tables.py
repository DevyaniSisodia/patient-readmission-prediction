import sqlite3
import os
import pandas as pd
import random

# Ensure the data folder exists
os.makedirs("../data", exist_ok=True)

# Connect to SQLite DB
conn = sqlite3.connect("../data/patient_readmission.db")
cursor = conn.cursor()

# Create the table
cursor.execute("""
CREATE TABLE IF NOT EXISTS admission_features (
    patient_id INTEGER PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    time_in_hospital INTEGER,
    num_lab_procedures INTEGER,
    readmitted INTEGER
)
""")

# Clear existing data (optional, for re-runs)
cursor.execute("DELETE FROM admission_features")

# Generate 100 synthetic records
sample_data = []
for i in range(1, 101):
    age = random.randint(20, 90)
    gender = random.choice(['Male', 'Female'])
    time_in_hospital = random.randint(1, 14)
    num_lab_procedures = random.randint(10, 80)
    readmitted = random.choice([0, 1])
    sample_data.append((i, age, gender, time_in_hospital, num_lab_procedures, readmitted))

# Insert the data
cursor.executemany("""
INSERT INTO admission_features (patient_id, age, gender, time_in_hospital, num_lab_procedures, readmitted)
VALUES (?, ?, ?, ?, ?, ?)
""", sample_data)

# Commit and close
conn.commit()
conn.close()

print("✅ Inserted 100 synthetic records into admission_features table.")
