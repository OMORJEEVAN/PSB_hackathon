import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

# -------- Cash Flow --------
income = np.random.randint(20000, 100000, n)
expense = income * np.random.uniform(0.6, 1.2, n)

savings_ratio = (income - expense) / income
income_stability = np.random.uniform(0.4, 0.95, n)
expense_volatility = np.random.uniform(0.1, 0.8, n)

cashflow_df = pd.DataFrame({
    "user_id": range(1, n+1),
    "monthly_income": income,
    "monthly_expense": expense.astype(int),
    "savings_ratio": savings_ratio.round(2),
    "income_stability": income_stability.round(2),
    "expense_volatility": expense_volatility.round(2)
})

# -------- Geolocation --------
location_changes = np.random.randint(1, 20, n)
distance = np.random.randint(1, 100, n)
variance = np.random.uniform(0.05, 0.9, n)

stability_score = 1 - (variance + location_changes/20)/2

geo_df = pd.DataFrame({
    "user_id": range(1, n+1),
    "num_location_changes": location_changes,
    "avg_distance_km": distance,
    "location_variance": variance.round(2),
    "stability_score": stability_score.round(2)
})

# -------- SAVE FILES --------
cashflow_df.to_csv("raw/cashflow_data.csv", index=False)
geo_df.to_csv("raw/geolocation_data.csv", index=False)

print("CSV files saved successfully!")