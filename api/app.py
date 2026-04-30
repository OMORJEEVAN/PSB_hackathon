from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import numpy as np
import pickle
import hashlib
import traceback
from typing import Optional

app = FastAPI(title="AI Credit Scoring")


# -------------------------------
# LOAD MODELS
# -------------------------------
lgb_base = pickle.load(open("../saved_models/lgbm_model.pkl", "rb"))
xgb_base = pickle.load(open("../saved_models/xgb_model.pkl", "rb"))
nn_model = pickle.load(open("../saved_models/nn_model.pkl", "rb"))
lr_model = pickle.load(open("../saved_models/logistic_model.pkl", "rb"))

scaler_lr = pickle.load(open("../saved_models/scaler.pkl", "rb"))
scaler_nn = pickle.load(open("../saved_models/nn_scaler.pkl", "rb"))

lgb_meta = pickle.load(open("../saved_models/lgb_meta.pkl", "rb"))
xgb_meta = pickle.load(open("../saved_models/xgb_meta.pkl", "rb"))


# -------------------------------
# FEATURE SETS (FROM YOUR TRAINING)
# -------------------------------

# LGB/XGB features
TREE_DROP = [
    "user_id","target",
    "risk_taking_score","avg_delay_days",
    "on_time_payment_ratio","composite_risk_score","phone_risk_score"
]

# NN drop
NN_DROP = ["user_id","target","risk_taking_score"]

# LR drop
LR_DROP = ["user_id","target"]

# stacking drop
STACK_DROP = ["user_id","target","risk_taking_score"]


# -------------------------------
# CONSENT GROUPS
# -------------------------------
FEATURE_GROUPS = {
    "phone": ["phone_payment_consistency_score","recharge_frequency","phone_risk_score"],
    "ecommerce": ["purchase_frequency","avg_order_value","return_ratio"],
    "psychometric": ["conscientiousness_score","risk_taking_score","financial_discipline_score"],
    "financial": ["monthly_income","monthly_expense","saving_ratio","income_stability","expense_volatility"],
    "geolocation": ["num_location_changes","avg_distance_km","location_variance","stability_score"],
    "credit": ["on_time_payment_ratio","avg_delay_days"],
    "merchant": ["merchant_rating_score","high_rating_ratio","trusted_vendor_ratio"]
}


# -------------------------------
# HELPERS
# -------------------------------
def hash_user_id(uid):
    return hashlib.sha256(str(uid).encode()).hexdigest()


def apply_consent(df, flags):
    for g, cols in FEATURE_GROUPS.items():
        if not flags.get(g, True):
            for c in cols:
                if c in df.columns:
                    df[c] = 0
    return df


def prepare(df, drop_cols):
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def explain(row):
    reasons = []

    if row.get("saving_ratio",0) > 0.3:
        reasons.append("Good savings")

    if row.get("avg_delay_days",0) > 5:
        reasons.append("Payment delays")

    if row.get("stability_score",0) > 0.7:
        reasons.append("Stable location")

    return reasons


# -------------------------------
# API
# -------------------------------
@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(...),
    phone: Optional[bool] = Form(True),
    ecommerce: Optional[bool] = Form(True),
    psychometric: Optional[bool] = Form(True),
    financial: Optional[bool] = Form(True),
    geolocation: Optional[bool] = Form(True),
    credit: Optional[bool] = Form(True),
    merchant: Optional[bool] = Form(True),
):
    try:
        df = pd.read_csv(file.file)

        if df.empty:
            raise ValueError("Empty CSV")

        consent_flags = {
            "phone": phone,
            "ecommerce": ecommerce,
            "psychometric": psychometric,
            "financial": financial,
            "geolocation": geolocation,
            "credit": credit,
            "merchant": merchant
        }

        df = apply_consent(df, consent_flags)

        # -------------------------------
        # BASE MODELS (DIFFERENT INPUTS)
        # -------------------------------

        X_tree = prepare(df, TREE_DROP)
        X_lr = prepare(df, LR_DROP)
        X_nn = prepare(df, NN_DROP)
        X_stack = prepare(df, STACK_DROP)

        # scale
        X_lr_scaled = scaler_lr.transform(X_lr)
        X_nn_scaled = scaler_nn.transform(X_nn)

        # predictions
        p1 = lgb_base.predict_proba(X_tree)[:,1]
        p2 = xgb_base.predict_proba(X_tree)[:,1]
        p3 = nn_model.predict_proba(X_nn_scaled)[:,1]
        p4 = lr_model.predict_proba(X_lr_scaled)[:,1]

        # -------------------------------
        # STACKING
        # -------------------------------
        stacked = pd.DataFrame({
            "lgb": p1,
            "xgb": p2,
            "nn": p3,
            "lr": p4
        })

        m1 = lgb_meta.predict_proba(stacked)[:,1]
        m2 = xgb_meta.predict_proba(stacked)[:,1]

        final = (m1 + m2)/2

        # -------------------------------
        # OUTPUT
        # -------------------------------
        results = []
        for i in range(len(df)):
            score = float(final[i])

            label = "LOW" if score<0.3 else "MEDIUM" if score<0.7 else "HIGH"

            results.append({
                "user_id": hash_user_id(df.iloc[i].get("user_id",i)),
                "risk_score": score,
                "risk_category": label,
                "explanation": explain(df.iloc[i])
            })

        return {
            "status":"success",
            "results":results,
            "consent_used":consent_flags
        }

    except Exception as e:
        return {
            "status":"error",
            "message":str(e),
            "trace":traceback.format_exc()
        }