import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
try:
    model = joblib.load("model/best_model.pkl")
    st.success("✅ Model loaded successfully.")
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'best_model.pkl' exists in 'model/' folder.")
    st.stop()

st.title("🏥 Patient Readmission Risk Predictor")

# User inputs
age = st.slider("🧓 Age", 20, 90, 50)
gender = st.selectbox("🧬 Gender", ["Male", "Female"])
time_in_hospital = st.slider("🏨 Days in Hospital", 1, 14, 5)
num_lab_procedures = st.slider("🧪 Lab Procedures", 10, 80, 40)
num_medications = st.slider("💊 Number of Medications", 1, 30, 10)
number_diagnoses = st.slider("📋 Number of Diagnoses", 1, 9, 4)
num_procedures = st.slider("🔧 Number of Procedures", 0, 6, 2)
admission_type_id = st.selectbox("🏥 Admission Type ID", list(range(1, 9)))
discharge_disposition_id = st.selectbox("🏁 Discharge Disposition ID", list(range(1, 31)))
race = st.selectbox("🌎 Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
A1Cresult = st.selectbox("🩸 A1C Result", ["None", "Norm", ">7", ">8"])
insulin = st.selectbox("💉 Insulin", ["No", "Up", "Down", "Steady"])
change = st.selectbox("📈 Change in Medication", ["No", "Ch"])

# Map/encode inputs to match training format
input_dict = {
    'age': age,
    'gender': 1 if gender == 'Female' else 0,
    'time_in_hospital': time_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_medications': num_medications,
    'number_diagnoses': number_diagnoses,
    'num_procedures': num_procedures,
    'admission_type_id': admission_type_id,
    'discharge_disposition_id': discharge_disposition_id,
    'race_' + race: 1,
    'A1Cresult_' + A1Cresult: 1,
    'insulin_' + insulin: 1,
    'change_' + change: 1,
}

# Create full input vector with all expected columns
# You must ensure this list matches the one used during training after `pd.get_dummies`
expected_features = model.get_booster().feature_names

# Initialize with zeros
input_data = pd.DataFrame([np.zeros(len(expected_features))], columns=expected_features)

# Fill in the provided values
for key, val in input_dict.items():
    if key in input_data.columns:
        input_data.at[0, key] = val
    elif key in ['age', 'gender', 'time_in_hospital', 'num_lab_procedures',
                 'num_medications', 'number_diagnoses', 'num_procedures',
                 'admission_type_id', 'discharge_disposition_id']:
        input_data.at[0, key] = val

# Prediction
if st.button("🔍 Predict Readmission"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.warning(f"🔁 High Risk of Readmission (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Readmission (Probability: {prob:.2f})")
