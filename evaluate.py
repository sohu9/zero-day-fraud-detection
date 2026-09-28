import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from data_pipeline import add_time_features, ID_COLUMNS, TARGET_COLUMN


MODEL_FILE = "zero_day_autoencoder.keras"
PREPROCESSOR_FILE = "preprocessor.joblib"
THRESHOLD_FILE = "threshold.joblib"
TEST_FILE = "data/transactions_test.csv"


def evaluate_model():

    print("🔍 [1/5] Loading trained model...")

    model = load_model(MODEL_FILE)

    print("✅ Model loaded.")

    # ---------------------------------------------------------
    # Load SAME preprocessing fitted on training data
    # ---------------------------------------------------------

    print("\n🔧 [2/5] Loading training preprocessing artifact...")

    artifact = joblib.load(PREPROCESSOR_FILE)

    preprocessor = artifact["preprocessor"]
    feature_names = artifact["feature_names"]

    print(
        f"✅ Preprocessor loaded with "
        f"{len(feature_names)} features."
    )

    # ---------------------------------------------------------
    # Load frozen threshold
    # ---------------------------------------------------------

    threshold_data = joblib.load(THRESHOLD_FILE)

    threshold = float(
        threshold_data["threshold"]
    )

    print(
        f"✅ Frozen threshold: {threshold:.6f}"
    )

    # ---------------------------------------------------------
    # Load FUTURE test data
    # ---------------------------------------------------------

    print("\n📊 [3/5] Loading future test dataset...")

    df = pd.read_csv(TEST_FILE)

    print(
        f"Test dataset: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    # Preserve IDs separately
    transaction_ids = (
        df["transaction_id"].copy()
        if "transaction_id" in df.columns
        else None
    )

    customer_ids = (
        df["customer_id"].copy()
        if "customer_id" in df.columns
        else None
    )

    merchant_ids = (
        df["merchant_id"].copy()
        if "merchant_id" in df.columns
        else None
    )

    # Target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "is_fraud column not found in test dataset."
        )

    y = df[TARGET_COLUMN].to_numpy()

    # ---------------------------------------------------------
    # Prepare test features
    # ---------------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    # IDs are preserved separately but not sent to model
    X = X.drop(
        columns=[
            col for col in ID_COLUMNS
            if col in X.columns
        ]
    )

    # Same time engineering as training
    X = add_time_features(X)

    # IMPORTANT:
    # transform only — DO NOT fit again
    X_processed = preprocessor.transform(X)

    X_processed = np.asarray(
        X_processed,
        dtype=np.float32
    )

    print(
        f"✅ Test data transformed using "
        f"training preprocessing."
    )

    print(
        f"Model input shape: {X_processed.shape}"
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    print(
        "\n⚙️ [4/5] Calculating reconstruction errors..."
    )

    predictions = model.predict(
        X_processed,
        verbose=0
    )

    mse_errors = np.mean(
        np.square(
            X_processed - predictions
        ),
        axis=1
    )

    # ---------------------------------------------------------
    # Fraud classification
    # ---------------------------------------------------------

    y_pred = (
        mse_errors > threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y,
        y_pred,
        labels=[0, 1]
    )

    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("📈 ZERO-DAY FRAUD DETECTION — FUTURE TEST RESULTS")
    print("=" * 60)

    print(
        f"Total transactions : {len(y)}"
    )

    print(
        f"Actual genuine     : {(y == 0).sum()}"
    )

    print(
        f"Actual fraud       : {(y == 1).sum()}"
    )

    print(
        f"Threshold          : {threshold:.6f}"
    )

    print("\nConfusion Matrix")
    print(
        "                 Predicted"
    )
    print(
        "                 Genuine  Fraud"
    )
    print(
        f"Actual Genuine   {cm[0,0]:8d} {cm[0,1]:7d}"
    )
    print(
        f"Actual Fraud     {cm[1,0]:8d} {cm[1,1]:7d}"
    )

    print("\nPerformance Metrics")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    # ---------------------------------------------------------
    # Reconstruction error analysis
    # ---------------------------------------------------------

    genuine_errors = mse_errors[y == 0]
    fraud_errors = mse_errors[y == 1]

    print("\nReconstruction Error")

    print(
        f"Genuine mean : {genuine_errors.mean():.6f}"
    )

    print(
        f"Fraud mean   : {fraud_errors.mean():.6f}"
    )

    print(
        f"Genuine median : "
        f"{np.median(genuine_errors):.6f}"
    )

    print(
        f"Fraud median   : "
        f"{np.median(fraud_errors):.6f}"
    )

    # ---------------------------------------------------------
    # Detection counts
    # ---------------------------------------------------------

    detected_fraud = int(
        (y_pred == 1).sum()
    )

    missed_fraud = int(
        ((y == 1) & (y_pred == 0)).sum()
    )

    false_alerts = int(
        ((y == 0) & (y_pred == 1)).sum()
    )

    print("\nDetection Summary")

    print(
        f"Fraud detected : {detected_fraud}"
    )

    print(
        f"Fraud missed   : {missed_fraud}"
    )

    print(
        f"False alerts   : {false_alerts}"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Save evaluation results
    # ---------------------------------------------------------

    results = {
        "threshold": threshold,
        "total_transactions": len(y),
        "actual_fraud": int((y == 1).sum()),
        "actual_genuine": int((y == 0).sum()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fraud_detected": detected_fraud,
        "fraud_missed": missed_fraud,
        "false_alerts": false_alerts,
        "genuine_mean_error": float(genuine_errors.mean()),
        "fraud_mean_error": float(fraud_errors.mean())
    }

    joblib.dump(
        results,
        "evaluation_results.joblib"
    )

    print(
        "\n💾 Saved evaluation_results.joblib"
    )

    print(
        "\n🎉 Evaluation completed successfully."
    )


if __name__ == "__main__":
    evaluate_model()