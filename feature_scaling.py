import pandas as pd
from sklearn.preprocessing import StandardScaler

print("1. Loading dataset...")

df = pd.read_csv("data/transactions_train.csv")

print("Original Shape:", df.shape)

df = df.drop(columns=["transaction_id"])

numeric_columns = [
    "account_age_days",
    "avg_monthly_spend",
    "merchant_risk_score",
    "transaction_amount",
    "ip_risk_score",
    "txn_count_1h",
    "txn_count_24h",
    "failed_txn_count_24h",
    "geo_distance_from_last_txn",
    "amount_deviation_from_user_mean",
    "post_auth_risk_score"
]

scaler = StandardScaler()

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("2. Feature Scaling Completed")

print("\nScaled Features:")
print(df[numeric_columns].head())

df.to_csv("data/scaled_transactions_train.csv", index=False)

print("\nScaled dataset saved successfully.")