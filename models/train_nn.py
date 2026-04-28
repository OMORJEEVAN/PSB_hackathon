import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

from sklearn.neural_network import MLPClassifier
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
X = df.drop(columns=["user_id", "target","risk_taking_score"])  ## AUC Score: 0.99, Accuracy: 0.99 leakiest features removed
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
# 4. FEATURE SCALING (CRITICAL)
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed")


# -------------------------------
# 5. TRAIN NEURAL NETWORK
# -------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # 2 layers
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Neural Network trained")


# -------------------------------
# 6. EVALUATION
# -------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nMODEL PERFORMANCE (Neural Network)")
print("AUC Score:", round(auc, 4))
print("Accuracy:", round(accuracy, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------
# 7. SAVE MODEL + SCALER
# -------------------------------
with open("../saved_models/nn_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../saved_models/nn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nNeural Network model saved!")