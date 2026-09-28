import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline


TARGET_COLUMN = "is_fraud"

# IDs are identifiers, not numerical behavioral features.
ID_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id"
]


def add_time_features(df):
    """
    Convert transaction_time into meaningful numerical features.
    Raw timestamp is not passed directly to the model.
    """

    df = df.copy()

    if "transaction_time" in df.columns:

        dt = pd.to_datetime(
            df["transaction_time"],
            errors="coerce"
        )

        if dt.isna().any():
            raise ValueError(
                "Invalid transaction_time value found."
            )

        # Hour as continuous value: 0.0 - 23.999...
        df["transaction_hour"] = (
            dt.dt.hour
            + dt.dt.minute / 60.0
            + dt.dt.second / 3600.0
        )

        # Monday = 0 ... Sunday = 6
        df["transaction_day_of_week"] = dt.dt.dayofweek

        # Weekend = 1, weekday = 0
        df["is_weekend"] = (
            dt.dt.dayofweek >= 5
        ).astype(int)

        # Raw timestamp removed
        df = df.drop(columns=["transaction_time"])

    return df


def build_preprocessor(df):

    df = df.copy()

    # Remove IDs from ML features
    df = df.drop(
        columns=[
            col for col in ID_COLUMNS
            if col in df.columns
        ]
    )

    # Remove target
    if TARGET_COLUMN in df.columns:
        X = df.drop(columns=[TARGET_COLUMN])
    else:
        X = df.copy()

    # Convert timestamp
    X = add_time_features(X)

    # Find categorical features
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # Remaining features are numerical
    numerical_columns = [
        col for col in X.columns
        if col not in categorical_columns
    ]

    print("\nCategorical features:")
    for col in categorical_columns:
        print(" -", col)

    print("\nNumerical features:")
    for col in numerical_columns:
        print(" -", col)

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            )
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return (
        preprocessor,
        numerical_columns,
        categorical_columns
    )


def run_pipeline(
    data_path="data/transactions_train.csv",
    save_preprocessor=True
):

    print("🚀 Data Pipeline Started...")

    df = pd.read_csv(data_path)

    print(
        f"✅ Dataset loaded: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' column not found."
        )

    y = df[TARGET_COLUMN].copy()

    # Build preprocessing using training data
    (
        preprocessor,
        numerical_columns,
        categorical_columns
    ) = build_preprocessor(df)

    # Prepare X
    X = df.drop(columns=[TARGET_COLUMN])

    # Remove IDs only from model input
    X = X.drop(
        columns=[
            col for col in ID_COLUMNS
            if col in X.columns
        ]
    )

    # Time engineering
    X = add_time_features(X)

    print("\n🔧 Fitting preprocessing on TRAINING data...")

    X_processed = preprocessor.fit_transform(X)

    X_processed = np.asarray(
        X_processed,
        dtype=np.float32
    )

    # ColumnTransformer output order:
    # numerical first, categorical second
    feature_names = (
        numerical_columns +
        categorical_columns
    )

    print(
        f"✅ Preprocessing complete."
    )

    print(
        f"✅ Final feature count: "
        f"{X_processed.shape[1]}"
    )

    if save_preprocessor:

        artifact = {
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns
        }

        joblib.dump(
            artifact,
            "preprocessor.joblib"
        )

        print(
            "💾 Saved: preprocessor.joblib"
        )

    return X_processed, y, feature_names


if __name__ == "__main__":

    X_train, y_train, feature_names = run_pipeline()

    print("\n" + "=" * 50)
    print("PIPELINE TEST RESULT")
    print("=" * 50)

    print("Processed X shape :", X_train.shape)
    print("Target shape      :", y_train.shape)
    print("Feature count     :", len(feature_names))

    print(
        "Genuine count     :",
        int((y_train == 0).sum())
    )

    print(
        "Fraud count       :",
        int((y_train == 1).sum())
    )

    print("\nModel features:")

    for i, name in enumerate(feature_names):
        print(f"{i + 1}. {name}")

    print("\n🎉 Pipeline test completed.")