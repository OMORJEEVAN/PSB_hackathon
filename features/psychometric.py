def compute_psychometric_scores(df):

    csn_cols = [f"CSN{i}_E" for i in range(1, 11)]
    est_cols = [f"EST{i}_E" for i in range(1, 11)]
    agr_cols = [f"AGR{i}_E" for i in range(1, 11)]

    df["conscientiousness_score"] = df[csn_cols].mean(axis=1)

    df["risk_taking_score"] = 6 - df[est_cols].mean(axis=1)

    df["financial_discipline_score"] = (
        0.7 * df[csn_cols].mean(axis=1) +
        0.3 * df[agr_cols].mean(axis=1)
    )

    # Ensure user_id exists
    if "user_id" not in df.columns:
        df["user_id"] = df.index

    return df[[
        "user_id",
        "conscientiousness_score",
        "risk_taking_score",
        "financial_discipline_score"
    ]]