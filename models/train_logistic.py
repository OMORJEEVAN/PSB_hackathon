import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report


# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("../data/processed/main_data.csv")

print("Data loaded successfully")
print("Shape:", df.shape)


# -------------------------------
# 2. SEPARATE FEATURES & TARGET
# -------------------------------
X = df.drop(columns=["user_id", "target"])
y = df["target"]

print("Features and target separated")


# -------------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split done")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# -------------------------------
# 4. FEATURE SCALING
# -------------------------------
scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

print(" Feature scaling completed")


# -------------------------------
# 5. TRAIN LOGISTIC REGRESSION
# -------------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print(" Logistic Regression trained")


# -------------------------------
# 6. EVALUATION
# -------------------------------
# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n MODEL PERFORMANCE")
print("AUC Score:", round(auc, 4))
print("Accuracy:", round(accuracy, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------
# 7. FEATURE IMPORTANCE (COEFFICIENTS)
# -------------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

# Sort by importance
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

print("\n Top Positive Features (LOW RISK):")
print(coefficients.head(10))

print("\n Top Negative Features (HIGH RISK):")
print(coefficients.tail(10))


# -------------------------------
# SAVE MODEL (optional but useful)
# -------------------------------
import pickle

with open("../saved_models/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../saved_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n Model and scaler saved!")