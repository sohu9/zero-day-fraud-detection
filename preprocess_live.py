import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Training dataset se preprocessing information prepare karna
def prepare_preprocessor(data_path="data/transactions_train.csv"):

    df = pd.read_csv(data_path)

    # Same columns jo existing data_pipeline.py mein drop hoti hain
    columns_to_drop = ["transaction_id", "user_name"]
    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns]
    )

    # Target remove
    if "is_fraud" in df.columns:
        X = df.drop(columns=["is_fraud"])
    else:
        X = df.copy()

    # Encoders store karenge
    encoders = {}

    # Existing pipeline ke according categorical columns
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder

    # Same StandardScaler approach
    scaler = StandardScaler()
    scaler.fit(X)

    return encoders, scaler, X.columns.tolist()


# Live transaction ko model-ready banana
def preprocess_transaction(data, encoders, scaler, feature_columns):

    df = pd.DataFrame([data])

    # Same feature order maintain karna
    df = df[feature_columns]

    # Categorical values encode karna
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    # StandardScaler apply karna
    scaled = scaler.transform(df)

    return scaled