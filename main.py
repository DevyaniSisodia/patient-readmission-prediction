# save this as main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model/best_model.pkl")

class PatientData(BaseModel):
    age: int
    gender: str
    time_in_hospital: int
    num_lab_procedures: int

@app.post("/predict")
def predict(data: PatientData):
    gender_num = 1 if data.gender.lower() == 'female' else 0
    features = [[data.age, gender_num, data.time_in_hospital, data.num_lab_procedures]]
    prediction = model.predict(features)[0]
    return {"readmitted": int(prediction)}
