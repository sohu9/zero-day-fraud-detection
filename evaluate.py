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

    # ---------------------------------------------------------
    # Preserve IDs separately
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

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
    # Transform only — DO NOT fit again
    X_processed = preprocessor.transform(X)

    X_processed = np.asarray(
        X_processed,
        dtype=np.float32
    )

    print(
        "✅ Test data transformed using "
        "training preprocessing."
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
    # Extract Confusion Matrix Values
    # ---------------------------------------------------------

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("📈 ZERO-DAY FRAUD DETECTION — FUTURE TEST RESULTS")
    print("=" * 70)

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

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        "                 Predicted"
    )

    print(
        "                 Genuine   Fraud"
    )

    print(
        f"Actual Genuine   {tn:8d} {fp:8d}"
    )

    print(
        f"Actual Fraud     {fn:8d} {tp:8d}"
    )

    print("\nConfusion Matrix Explanation:")
    print(
        f"TN (True Negative)  : {tn} "
        f"→ Genuine transactions correctly identified as genuine."
    )

    print(
        f"FP (False Positive) : {fp} "
        f"→ Genuine transactions incorrectly flagged as fraud."
    )

    print(
        f"FN (False Negative) : {fn} "
        f"→ Fraud transactions missed by the model."
    )

    print(
        f"TP (True Positive)  : {tp} "
        f"→ Fraud transactions correctly detected by the model."
    )

    # ---------------------------------------------------------
    # Performance Metrics
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)

    print(
        f"Accuracy  : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall    : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score  : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    # ---------------------------------------------------------
    # Automatically Generated Metric Reasons
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("WHAT DO THESE RESULTS MEAN?")
    print("-" * 70)

    print(
        f"Accuracy Reason : "
        f"The model correctly classified {tn + tp} "
        f"out of {len(y)} total transactions."
    )

    print(
        f"Precision Reason : "
        f"Out of {tp + fp} transactions predicted as fraud, "
        f"{tp} were actually fraud."
    )

    print(
        f"Recall Reason : "
        f"Out of {(y == 1).sum()} actual fraud transactions, "
        f"the model detected {tp} and missed {fn}."
    )

    print(
        f"F1 Score Reason : "
        f"F1-score combines precision and recall into a single "
        f"balance measure."
    )

    # ---------------------------------------------------------
    # Fraud Detection Interpretation
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("FRAUD DETECTION INTERPRETATION")
    print("-" * 70)

    if recall >= 0.90:
        print(
            "✅ High Recall: "
            "The model detected a large proportion of the "
            "actual fraud transactions."
        )
    else:
        print(
            "⚠️ Recall Observation: "
            "Some actual fraud transactions were not detected."
        )

    if precision >= 0.80:
        print(
            "✅ Precision Observation: "
            "Most transactions flagged as fraud were actually fraud."
        )
    else:
        print(
            "⚠️ Precision Observation: "
            "Some genuine transactions were also flagged as fraud."
        )

    print(
        f"\nActual fraud transactions : {(y == 1).sum()}"
    )

    print(
        f"Correctly detected fraud  : {tp}"
    )

    print(
        f"Missed fraud              : {fn}"
    )

    print(
        f"False fraud alerts         : {fp}"
    )

    # ---------------------------------------------------------
    # Reconstruction Error Analysis
    # ---------------------------------------------------------

    genuine_errors = mse_errors[y == 0]
    fraud_errors = mse_errors[y == 1]

    print("\n" + "-" * 70)
    print("RECONSTRUCTION ERROR ANALYSIS")
    print("-" * 70)

    print(
        f"Genuine mean   : "
        f"{genuine_errors.mean():.6f}"
    )

    print(
        f"Fraud mean     : "
        f"{fraud_errors.mean():.6f}"
    )

    print(
        f"Genuine median : "
        f"{np.median(genuine_errors):.6f}"
    )

    print(
        f"Fraud median   : "
        f"{np.median(fraud_errors):.6f}"
    )

    print(
        f"\nThreshold      : "
        f"{threshold:.6f}"
    )

    print(
        "Decision rule  : "
        "Reconstruction Error > Threshold → Anomaly/Fraud Flag"
    )

    # ---------------------------------------------------------
    # Detection Counts
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

    print("\n" + "-" * 70)
    print("DETECTION SUMMARY")
    print("-" * 70)

    print(
        f"Fraud detected : {detected_fraud}"
    )

    print(
        f"Fraud missed   : {missed_fraud}"
    )

    print(
        f"False alerts   : {false_alerts}"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # Save Evaluation Results
    # ---------------------------------------------------------

    results = {
        "threshold": threshold,

        "total_transactions": len(y),

        "actual_fraud": int(
            (y == 1).sum()
        ),

        "actual_genuine": int(
            (y == 0).sum()
        ),

        # Confusion Matrix
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,

        # Metrics
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

        # Detection
        "fraud_detected": detected_fraud,
        "fraud_missed": missed_fraud,
        "false_alerts": false_alerts,

        # Reconstruction Error
        "genuine_mean_error": float(
            genuine_errors.mean()
        ),

        "fraud_mean_error": float(
            fraud_errors.mean()
        ),

        "genuine_median_error": float(
            np.median(genuine_errors)
        ),

        "fraud_median_error": float(
            np.median(fraud_errors)
        ),
        "genuine_errors": genuine_errors,
        "fraud_errors": fraud_errors
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