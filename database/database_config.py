# database/database.config.py
from sqlalchemy import create_engine

def get_engine():
    return create_engine("sqlite:///../data/patient_readmission.db")  # or your full path
