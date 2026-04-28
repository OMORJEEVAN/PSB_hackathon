import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

# Total transactions per user
num_transactions = np.random.randint(5, 200, n)

# Average merchant rating (biased toward 3–5)
avg_rating = np.random.normal(loc=4.0, scale=0.5, size=n)
avg_rating = np.clip(avg_rating, 1, 5)

# % of purchases from highly rated merchants (>=4 stars)
high_rating_ratio = np.random.uniform(0.3, 1.0, n)

# Trusted vendor ratio (correlated with high_rating_ratio)
trusted_vendor_ratio = (
    high_rating_ratio * 0.7 + np.random.uniform(0.0, 0.3, n)
)
trusted_vendor_ratio = np.clip(trusted_vendor_ratio, 0, 1)

# Create DataFrame
merchant_df = pd.DataFrame({
    "user_id": range(1, n+1),
    "num_transactions": num_transactions,
    "avg_merchant_rating": avg_rating.round(2),
    "high_rating_ratio": high_rating_ratio.round(2),
    "trusted_vendor_ratio": trusted_vendor_ratio.round(2)
})

# Save to CSV
merchant_df.to_csv("merchant_ratings_data.csv", index=False)

print("Merchant ratings data saved successfully!")
print(merchant_df.head())