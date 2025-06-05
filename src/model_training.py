"""
Machine Learning Model Training Pipeline for Patient Readmission Prediction
Implements multiple ML algorithms with hyperparameter tuning to achieve 82%+ AUC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix, 
                           roc_curve, precision_recall_curve, average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

class ReadmissionPredictor:
    def __init__(self, random_state: int = 42):
        """
        Initialize the Patient Readmission 
        """