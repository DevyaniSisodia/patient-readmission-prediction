# save as app.py
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/best_model.pkl")

st.title("Patient Readmission Predictor")

age = st.slider("Age", 0, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
hospital_days = st.slider("Days in Hospital", 1, 14, 3)
labs = st.slider("Number of Lab Procedures", 1, 100, 40)

if st.button("Predict"):
    gender_num = 1 if gender == "Female" else 0
    input_data = np.array([[age, gender_num, hospital_days, labs]])
    prediction = model.predict(input_data)
    st.success("Readmitted" if prediction[0] else "Not Readmitted")
