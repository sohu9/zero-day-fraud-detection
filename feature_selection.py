import pandas as pd

print("1. Loading dataset...")

train_df = pd.read_csv("data/transactions_train.csv")

print("Original Shape:", train_df.shape)

print("\nOriginal Columns:")
print(train_df.columns.tolist())

drop_columns = [
    "transaction_id"
]

selected_df = train_df.drop(columns=drop_columns)

print("\n2. Feature Selection Completed")

print("Removed Columns:")
print(drop_columns)

print("\nNew Shape:", selected_df.shape)

print("\nRemaining Columns:")
print(selected_df.columns.tolist())