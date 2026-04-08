# ============================================
# CHRONIC KIDNEY DISEASE PREDICTION PROJECT
# ============================================

# ---------- IMPORT LIBRARIES ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# ---------- LOAD DATA ----------
df = pd.read_csv("kidney_disease.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# ---------- DROP ID COLUMN IF EXISTS ----------
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# ---------- CHECK DATA TYPES ----------
print("\nData Types:")
print(df.dtypes)

# ---------- CLEAN COLUMN NAMES ----------
df.columns = df.columns.str.strip()

# ---------- REPLACE TAB/SPACES IN DATA ----------
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(['\t?', '\tyes', '\tno', 'nan'], np.nan)

# ---------- FIX KNOWN CATEGORICAL ISSUES ----------
df.replace({
    'normal': 'normal',
    'abnormal': 'abnormal',
    'present': 'present',
    'notpresent': 'notpresent',
    'yes': 'yes',
    'no': 'no',
    'ckd\t': 'ckd',
    '\tno': 'no',
    '\tyes': 'yes'
}, inplace=True)

# ---------- CONVERT NUMERIC COLUMNS ----------
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
                'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ---------- HANDLE MISSING VALUES ----------
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ---------- ENCODE CATEGORICAL COLUMNS ----------
# Binary mapping based on CKD dataset structure
mapping_dict = {
    'rbc': {'normal': 0, 'abnormal': 1},
    'pc': {'normal': 0, 'abnormal': 1},
    'pcc': {'notpresent': 0, 'present': 1},
    'ba': {'notpresent': 0, 'present': 1},
    'htn': {'no': 0, 'yes': 1},
    'dm': {'no': 0, 'yes': 1},
    'cad': {'no': 0, 'yes': 1},
    'appet': {'poor': 0, 'good': 1},
    'pe': {'no': 0, 'yes': 1},
    'ane': {'no': 0, 'yes': 1},
    'class': {'ckd': 0, 'not ckd': 1}
}

for col, mapping in mapping_dict.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# ---------- HANDLE LEFTOVER CATEGORICAL DATA ----------
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.factorize(df[col])[0]

# ---------- FINAL CHECK ----------
print("\nCleaned Dataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df['class'].value_counts())

# ---------- OPTIONAL: DATA VISUALIZATION ----------
plt.figure(figsize=(8, 5))
sns.countplot(x='class', data=df)
plt.title("Class Distribution (0 = CKD, 1 = NOT CKD)")
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ---------- SPLIT FEATURES AND TARGET ----------
X = df.drop('class', axis=1)
y = df['class']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ---------- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ---------- MODEL TRAINING ----------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

results = {}

print("\n========== MODEL ACCURACY ==========\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# ---------- BEST MODEL ----------
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\n====================================")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print("====================================")

best_model = models[best_model_name]

# ---------- EVALUATE BEST MODEL ----------
y_pred_best = best_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# ---------- ACCURACY BAR CHART ----------
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# ---------- FINAL PREDICTION ----------
# IMPORTANT:
# Make sure values are in same order as dataset columns (except 'class')

print("\nFeature Columns Order:")
print(list(X.columns))

# Example patient data (replace with actual values based on your dataset columns)
# Update this according to your actual dataset column order
sample_data = [[
    48.0,   # age
    70.0,   # bp
    1.005,  # sg
    4.0,    # al
    0.0,    # su
    1,      # rbc
    0,      # pc
    1,      # pcc
    0,      # ba
    117.0,  # bgr
    56.0,   # bu
    3.8,    # sc
    111.0,  # sod
    2.5,    # pot
    11.2,   # hemo
    32.0,   # pcv
    6700.0, # wc
    3.9,    # rc
    1,      # htn
    0,      # dm
    0,      # cad
    1,      # appet
    1,      # pe
    1       # ane
]]

prediction = best_model.predict(sample_data)

print("\n========== FINAL PREDICTION ==========")
if prediction[0] == 0:
    print("The person is predicted to suffer from Chronic Kidney Disease (CKD).")
else:
    print("The person is predicted NOT to suffer from Chronic Kidney Disease (CKD).")
print("======================================")