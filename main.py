# Dataset Inspection + Preprocessing + Random Forest Training

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib


# 1. Load dataset

df = pd.read_csv("C:/Users/user/Desktop/AI task/plant_health_data.csv")

# 2. Initial Data Inspection

print("\n--- First 5 rows of Raw Dataset ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Column Information ---")
print(df.info())

print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

print("\n--- Statistical Summary ---")
print(df.describe())

# Drop non-informative columns
df = df.drop(['Timestamp', 'Plant_ID'], axis=1)

# -----------------------------
# 3. Convert target to binary: Healthy = 0, Diseased = 1
# Both Moderate Stress and High Stress -> Diseased
# -----------------------------
df['Plant_Health_Status_Binary'] = df['Plant_Health_Status'].apply(
    lambda x: 0 if x == 'Healthy' else 1
)


# 4. Separate features and target
TARGET_COL = 'Plant_Health_Status_Binary'
X = df.drop(['Plant_Health_Status', 'Plant_Health_Status_Binary'], axis=1)
y = df[TARGET_COL]


# 5. Normalize features

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\n--- First 5 rows of Normalized Features ---")
print(X_scaled_df.head())

print("\n--- Statistical Summary of Normalized Features ---")
print(X_scaled_df.describe())


# 6. Stratified train-test split (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)


# 7. Train Random Forest Classifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# 8. Save model, scaler, and datasets

joblib.dump(rf, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump((X_train, y_train), 'train_data.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')

print("\nTraining complete. Model and datasets saved successfully.")