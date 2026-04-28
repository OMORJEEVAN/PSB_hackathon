import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPClassifier


# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("../data/processed/main_data.csv")

X = df.drop(columns=["user_id", "target","risk_taking_score"])  ## leakiest features removed
y = df["target"]

print(" Data loaded")


# -------------------------------
# K-FOLD
# -------------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_nn = np.zeros(len(X))
oof_lr = np.zeros(len(X))


# -------------------------------
# LEVEL 0 MODELS
# -------------------------------
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n Fold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05)
    lgb_model.fit(X_train, y_train)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

    # XGB
    xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05,
                                 eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

    # Scaling for LR + NN
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Logistic
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_s, y_train)
    oof_lr[val_idx] = lr_model.predict_proba(X_val_s)[:, 1]

    # NN
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200)
    nn_model.fit(X_train_s, y_train)
    oof_nn[val_idx] = nn_model.predict_proba(X_val_s)[:, 1]


# -------------------------------
# LEVEL 1 DATASET
# -------------------------------
level1_X = pd.DataFrame({
    "lgb": oof_lgb,
    "xgb": oof_xgb,
    "nn": oof_nn,
    "lr": oof_lr
})

print("\n Level 1 dataset ready")


# -------------------------------
# LEVEL 1 MODELS
# -------------------------------

# Model 1: LightGBM on stacked features
lgb_meta = lgb.LGBMClassifier(n_estimators=200)
lgb_meta.fit(level1_X, y)
pred_lgb_meta = lgb_meta.predict_proba(level1_X)[:, 1]

# Model 2: XGBoost on stacked features
xgb_meta = xgb.XGBClassifier(n_estimators=200, eval_metric="logloss")
xgb_meta.fit(level1_X, y)
pred_xgb_meta = xgb_meta.predict_proba(level1_X)[:, 1]


# -------------------------------
# LEVEL 2 FINAL BLEND
# -------------------------------
final_pred = (pred_lgb_meta + pred_xgb_meta) / 2

auc = roc_auc_score(y, final_pred)

print("\n LEVEL 2 STACKING PERFORMANCE")
print("AUC Score:", round(auc, 4))


# -------------------------------
# SAVE MODELS
# -------------------------------
import pickle

with open("../saved_models/lgb_meta.pkl", "wb") as f:
    pickle.dump(lgb_meta, f)

with open("../saved_models/xgb_meta.pkl", "wb") as f:
    pickle.dump(xgb_meta, f)

print("\nLevel 2 stacking models saved!")