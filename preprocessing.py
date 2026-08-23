import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

print("1. Program started...")

# Load the training dataset
train_df = pd.read_csv('data/transactions_train.csv')
print("2. Training data loaded successfully!")

# 1. Encoding Categorical Columns
print("3. Starting label encoding for categorical columns...")
label_encoder = LabelEncoder()

# Identify columns that have text (object data type)
categorical_cols = train_df.select_dtypes(include=['object']).columns
print(f"Categorical columns found: {list(categorical_cols)}")

for col in categorical_cols:
    train_df[col] = label_encoder.fit_transform(train_df[col].astype(str))

print("4. Encoding completed successfully!")

# 2. Handling Imbalance using SMOTE
print("5. Handling class imbalance using SMOTE...")
target_column = 'is_fraud'

X = train_df.drop(columns=[target_column])
y = train_df[target_column]

print(f"Before SMOTE - Total rows: {X.shape[0]}, Fraud cases:py {y.sum()}")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"After SMOTE - Total rows: {X_resampled.shape[0]}, Fraud cases: {y_resampled.sum()}")
print("6. Preprocessing pipeline completed successfully!")