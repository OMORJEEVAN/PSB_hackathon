import numpy as np
import pandas as pd

np.random.seed(42)


def generate_phone_data(num_users=1000):
    """
    Generates synthetic phone bill payment behavior
    """

    user_ids = np.arange(1, num_users + 1)

    # Simulate number of bills per user
    total_bills = np.random.randint(6, 12, size=num_users)

    # On-time payments
    on_time = np.random.randint(3, total_bills + 1)

    phone_payment_consistency_score = on_time / total_bills

    # Delay in days (0 = on time)
    avg_bill_delay_days = np.random.exponential(scale=2, size=num_users)

    # Recharge frequency (monthly behavior)
    recharge_frequency = np.random.uniform(0.5, 1.5, size=num_users)

    df = pd.DataFrame({
        "user_id": user_ids,
        "phone_payment_consistency_score": phone_payment_consistency_score,
        "avg_bill_delay_days": avg_bill_delay_days,
        "recharge_frequency": recharge_frequency
    })

    return df


def main():
    df = generate_phone_data(num_users=1000)

    df.to_csv("raw/phone_data.csv", index=False)
    print("phone_data.csv created!")


if __name__ == "__main__":
    main()