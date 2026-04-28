import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPClassifier


# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("../data/processed/main_data.csv")

X = df.drop(columns=["user_id", "target","risk_taking_score"])  ## leakiest features removed
y = df["target"]

print("Data loaded")


# -------------------------------
# K-FOLD SETUP
# -------------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -------------------------------
# STORAGE FOR OUT-OF-FOLD PREDICTIONS
# -------------------------------
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_nn = np.zeros(len(X))
oof_lr = np.zeros(len(X))


# -------------------------------
# K-FOLD TRAINING
# -------------------------------
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # -------------------------------
    # LIGHTGBM
    # -------------------------------
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )
    lgb_model.fit(X_train, y_train)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

    # -------------------------------
    # XGBOOST
    # -------------------------------
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

    # -------------------------------
    # LOGISTIC REGRESSION (scaled)
    # -------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    oof_lr[val_idx] = lr_model.predict_proba(X_val_scaled)[:, 1]

    # -------------------------------
    # NEURAL NETWORK
    # -------------------------------
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200
    )
    nn_model.fit(X_train_scaled, y_train)
    oof_nn[val_idx] = nn_model.predict_proba(X_val_scaled)[:, 1]


# -------------------------------
# CREATE STACKED DATASET
# -------------------------------
stacked_X = pd.DataFrame({
    "lgb": oof_lgb,
    "xgb": oof_xgb,
    "nn": oof_nn,
    "lr": oof_lr
})

print("\nStacked features created")


# -------------------------------
# META MODEL (LEVEL 2)
# -------------------------------
meta_model = LogisticRegression()
meta_model.fit(stacked_X, y)

final_pred = meta_model.predict_proba(stacked_X)[:, 1]

auc = roc_auc_score(y, final_pred)

print("\nFINAL STACKED MODEL PERFORMANCE")
print("AUC Score:", round(auc, 4))


# -------------------------------
# SAVE META MODEL
# -------------------------------
import pickle

with open("../saved_models/stacking_model.pkl", "wb") as f:
    pickle.dump(meta_model, f)

print("\nStacking model saved!")