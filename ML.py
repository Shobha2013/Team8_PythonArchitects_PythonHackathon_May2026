#Import Prerequisites

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score, roc_auc_score, roc_curve
import xgboost as xgb

#Load Dataset

df = pd.read_csv(r"C:\Data Analyst_Resources\Python_Hackathon\HUPA-UC Diabetes Dataset\Practice_files\Cleaned_Diabetes_Data.csv")
demo = pd.read_csv(r"C:\Data Analyst_Resources\Python_Hackathon\HUPA-UC Diabetes Dataset\T1DM_patient_sleep_demographics_with_race.csv")

df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])

data = df.merge(demo, on="Patient_ID", how="left")

data.to_csv(r"C:\Data Analyst_Resources\Python_Hackathon\HUPA-UC Diabetes Dataset\Practice_files\Merged_Diabetes_Data.csv", index=False)

data.head()

# Create Target Variable

data["Hypoglycemia_Risk"] = (
    data["Blood_Glucose_mg_dl"] < 70
).astype(int)

#Select Features (Blood_Glucose_mg_dl removed to avoid data leakage)

features = [
    "Hour",
    "Minutes",
    "Basal_Insulin_Rate_Unit_hr",
    "Bolus_Insulin_Dose_Units",
    "Carbohydrate_Intake_Grams",
    "Step_count",
    "Calories_burned",
    "Heart_Rate_bpm",
    "Age"
]

X = data[features]

y = data["Hypoglycemia_Risk"]

# Check class balance
print("Class Distribution:\n", y.value_counts(normalize=True))

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL 1 — Logistic Regression
# Train Model

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train, y_train)

# Predict and Evaluate

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# Evaluation Metrics

print("Accuracy:", accuracy_score(y_test, lr_pred))

print("Precision:", precision_score(y_test, lr_pred))

print("Recall:", recall_score(y_test, lr_pred))

print("F1 Score:", f1_score(y_test, lr_pred))

print("ROC AUC:", roc_auc_score(y_test, lr_prob))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, lr_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")

plt.savefig(r"C:\Data Analyst_Resources\Python_Hackathon\HUPA-UC Diabetes Dataset\Practice_files\roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()