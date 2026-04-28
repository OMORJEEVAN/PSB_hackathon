import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

import lightgbm as lgb
import pickle


# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("../data/processed/main_data.csv")

print("Data loaded successfully")
print("Shape:", df.shape)


# -------------------------------
# 2. SEPARATE FEATURES & TARGET
# -------------------------------
drop_cols = [
    "user_id",
    "target",
    "risk_taking_score",   
    "avg_delay_days",
    "on_time_payment_ratio",
    "composite_risk_score",
    "phone_risk_score"
]

X = df.drop(columns=drop_cols)  ## AUC Score: 1.0, Accuracy: 0.9989 too leaking features removed
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


# -------------------------------
# 4. TRAIN LIGHTGBM MODEL
# -------------------------------
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)

print("LightGBM trained")


# -------------------------------
# 5. EVALUATION
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nMODEL PERFORMANCE (LightGBM)")
print("AUC Score:", round(auc, 4))
print("Accuracy:", round(accuracy, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------
# 6. FEATURE IMPORTANCE
# -------------------------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Important Features:")
print(importance_df.head(10))


# -------------------------------
# 7. SAVE MODEL
# -------------------------------
with open("../saved_models/lgbm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nLightGBM model saved!")