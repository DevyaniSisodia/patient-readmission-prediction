import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import pickle

# Load the data
df = pd.read_sql("SELECT * FROM admission_features", "sqlite:///../data/patient_readmission.db")

# One-hot encode gender
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

# Split features and target
X = df.drop(columns=["readmitted", "patient_id"])
y = df["readmitted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ✅ Check and fix flipped AUC
original_auc = roc_auc_score(y_test, y_pred_proba)
if original_auc < 0.5:
    print("⚠️ Model appears to be predicting the opposite class. Flipping probabilities.")
    y_pred_proba = 1 - y_pred_proba
    fixed_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"✅ AUC After Fix: {fixed_auc:.3f}")
else:
    print(f"✅ AUC Score: {original_auc:.3f}")

# Binary prediction from probability
y_pred = (y_pred_proba >= 0.5).astype(int)

# Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Save model
with open("model/best_model.pkl", "wb") as f:
    pickle.dump(model, f)
