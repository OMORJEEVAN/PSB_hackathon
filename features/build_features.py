import pandas as pd
from psychometric import compute_psychometric_scores


# -------------------------------
# 1. ONLINE RETAIL FEATURES
# -------------------------------
def build_retail_features(df):

    # Fix column name
    df.rename(columns={"CustomerID": "user_id"}, inplace=True)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Total orders per user
    orders = df.groupby("user_id")["InvoiceNo"].nunique()

    # Active days
    active_days = df.groupby("user_id")["InvoiceDate"].nunique()

    purchase_frequency = (orders / active_days).rename("purchase_frequency")

    # Avg order value
    df["total_price"] = df["Quantity"] * df["UnitPrice"]
    avg_order_value = df.groupby("user_id")["total_price"].mean().rename("avg_order_value")

    # Return ratio
    df["is_return"] = df["Quantity"] < 0
    return_ratio = df.groupby("user_id")["is_return"].mean().rename("return_ratio")

    retail_features = pd.concat(
        [purchase_frequency, avg_order_value, return_ratio], axis=1
    ).reset_index()

    return retail_features


# -------------------------------
# 2. repayment FEATURES
# -------------------------------
def build_credit_features(df):

    #Fix column name
    df.rename(columns={"ID": "user_id"}, inplace=True)

    pay_cols = [col for col in df.columns if "PAY_" in col]

    df["on_time_payment_ratio"] = (df[pay_cols] <= 0).sum(axis=1) / len(pay_cols)

    df["avg_delay_days"] = df[pay_cols].clip(lower=0).mean(axis=1)

    return df[["user_id", "on_time_payment_ratio", "avg_delay_days"]]


# -------------------------------
# 3. MERCHANT FEATURES
# -------------------------------
def build_merchant_features(df):

    return df[[
        "user_id",
        "avg_merchant_rating",
        "high_rating_ratio",
        "trusted_vendor_ratio"
    ]].rename(columns={
        "avg_merchant_rating": "merchant_rating_score"
    })

# -------------------------------
# 4. PHONE FEATURES
# -------------------------------
def build_phone_features(df):

    # Ensure correct column name
    df.rename(columns={"user_id": "user_id"}, inplace=True)

    df["avg_bill_delay_days_norm"] = df["avg_bill_delay_days"] / (df["avg_bill_delay_days"].max() + 1e-5)
    df["phone_risk_score"] = (
    0.7 * (1 - df["phone_payment_consistency_score"]) +
    0.3 * df["avg_bill_delay_days_norm"]
    )

    return df[[
        "user_id",
        "phone_payment_consistency_score",
        "avg_bill_delay_days",
        "recharge_frequency",
        "phone_risk_score"
    ]]


# -------------------------------
# 4. MAIN PIPELINE
# -------------------------------
def build_main_dataset():

    # Load all datasets
    retail = pd.read_csv("../data/raw/OnlineRetail.csv", encoding="ISO-8859-1")
    psych = pd.read_csv("../data/raw/data-final.csv", encoding="ISO-8859-1", sep="\t", nrows=50000)
    cashflow = pd.read_csv("../data/raw/cashflow_data.csv", encoding="ISO-8859-1")
    geo = pd.read_csv("../data/raw/geolocation_data.csv", encoding="ISO-8859-1")
    credit = pd.read_csv("../data/raw/UCI_Credit_Card.csv", encoding="ISO-8859-1")
    merchant = pd.read_csv("../data/raw/merchant_ratings_data.csv", encoding="ISO-8859-1")
    phone = pd.read_csv("../data/raw/phone_data.csv", encoding="ISO-8859-1")

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    retail_feat = build_retail_features(retail)
    psych_feat = compute_psychometric_scores(psych)
    credit_feat = build_credit_features(credit)
    merchant_feat = build_merchant_features(merchant)
    phone_feat = build_phone_features(phone)

    # -------------------------------
    # Merging
    # -------------------------------
    main_df = retail_feat

    main_df = main_df.merge(psych_feat, on="user_id", how="left")
    main_df = main_df.merge(cashflow, on="user_id", how="left")
    main_df = main_df.merge(geo, on="user_id", how="left")
    main_df = main_df.merge(credit_feat, on="user_id", how="left")
    main_df = main_df.merge(merchant_feat, on="user_id", how="left")
    main_df = main_df.merge(phone_feat, on="user_id", how="left")

    # -------------------------------
    # Derived Features
    # -------------------------------

    # Saving ratio
    main_df["saving_ratio"] = (
    (main_df["monthly_income"] - main_df["monthly_expense"]) /
    (main_df["monthly_income"] + 1e-5)
    )

    # Income stability (simple version)
    main_df["income_stability"] = main_df["monthly_income"] / (main_df["monthly_income"].max() + 1e-5)

    # Composite risk score
    main_df["composite_risk_score"] = (
    0.3 * main_df["phone_risk_score"] +
    0.3 * (1 - main_df["on_time_payment_ratio"]) +
    0.2 * main_df["avg_delay_days"] +
    0.2 * main_df["risk_taking_score"]
    )

    main_df = main_df.drop("avg_bill_delay_days", axis=1)
    # -------------------------------
    # Handle Missing Values
    # -------------------------------
    main_df.fillna(0, inplace=True)

    # -------------------------------
    # Create Meaningful Target
    # -------------------------------

    # Normalize key features first
    main_df["delay_norm"] = main_df["avg_delay_days"] / (main_df["avg_delay_days"].max() + 1e-5)

    # Risk score (domain-driven)
    main_df["risk_score"] = (
        (1 - main_df["saving_ratio"]) +
        main_df["delay_norm"] +
        (1 - main_df["on_time_payment_ratio"]) +
        (main_df["risk_taking_score"] / 5)
    )

    # Create binary target using median threshold
    threshold = main_df["risk_score"].median()

    main_df["target"] = (main_df["risk_score"] > threshold).astype(int)

    # Drop helper columns
    main_df.drop(["delay_norm", "risk_score"], axis=1, inplace=True)

        # -------------------------------
        # Save Final Dataset
        # -------------------------------
    main_df.to_csv("../data/processed/main_data.csv", index=False)

    print("main_data.csv created successfully!")

    return main_df


# -------------------------------
# RUN PIPELINE
# -------------------------------
if __name__ == "__main__":
    build_main_dataset()