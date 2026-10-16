import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model  # type: ignore
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_pipeline import add_time_features, ID_COLUMNS, TARGET_COLUMN


MODEL_FILE = "zero_day_autoencoder.keras"
PREPROCESSOR_FILE = "preprocessor.joblib"
THRESHOLD_FILE = "threshold.joblib"
TEST_FILE = "data/transactions_test.csv"


def evaluate_model():
    print("🔍 [1/5] Loading trained model...")
    model = load_model(MODEL_FILE, compile=False)
    print("✅ Model loaded.")

    print("\n🔧 [2/5] Loading training preprocessing artifact...")
    artifact = joblib.load(PREPROCESSOR_FILE)
    preprocessor = artifact["preprocessor"]
    feature_names = artifact["feature_names"]
    expected_columns = (
        list(artifact["numerical_columns"]) + list(artifact["categorical_columns"])
    )
    print(f"✅ Preprocessor loaded with {len(feature_names)} features.")

    threshold_data = joblib.load(THRESHOLD_FILE)
    threshold = float(
        threshold_data["threshold"] if isinstance(threshold_data, dict) else threshold_data
    )
    print(f"✅ Frozen threshold: {threshold:.6f}")

    print("\n📊 [3/5] Loading future test dataset...")
    df = pd.read_csv(TEST_FILE)
    print(f"Test dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    if TARGET_COLUMN not in df.columns:
        raise ValueError("is_fraud column not found in test dataset.")

    y = df[TARGET_COLUMN].to_numpy()

    X = df.drop(columns=[TARGET_COLUMN])
    X = X.drop(columns=[col for col in ID_COLUMNS if col in X.columns])
    X = add_time_features(X)
    X = X[expected_columns]

    X_processed = np.asarray(preprocessor.transform(X), dtype=np.float32)
    print("✅ Test data transformed using training preprocessing.")
    print(f"Model input shape: {X_processed.shape}")

    print("\n⚙️ [4/5] Calculating reconstruction errors...")
    predictions = model.predict(X_processed, verbose=0)
    mse_errors = np.mean(np.square(X_processed - predictions), axis=1)

    y_pred = (mse_errors > threshold).astype(int)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, mse_errors)

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    total_transactions = int(len(y))
    actual_genuine = int((y == 0).sum())
    actual_fraud = int((y == 1).sum())
    predicted_fraud = int((y_pred == 1).sum())
    correctly_classified = tn + tp

    genuine_errors = mse_errors[y == 0]
    fraud_errors = mse_errors[y == 1]

    genuine_mean = float(genuine_errors.mean())
    fraud_mean = float(fraud_errors.mean())
    genuine_median = float(np.median(genuine_errors))
    fraud_median = float(np.median(fraud_errors))

    print("\n" + "=" * 70)
    print("📈 ZERO-DAY FRAUD DETECTION — FUTURE TEST RESULTS")
    print("=" * 70)
    print(f"Total transactions : {total_transactions}")
    print(f"Actual genuine     : {actual_genuine}")
    print(f"Actual fraud       : {actual_fraud}")
    print(f"Threshold          : {threshold:.6f}")

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    print("                 Predicted")
    print("                 Genuine   Fraud")
    print(f"Actual Genuine   {tn:7d} {fp:7d}")
    print(f"Actual Fraud     {fn:7d} {tp:7d}")

    print("\nConfusion Matrix Explanation:")
    print(f"TN (True Negative)  : {tn} → Genuine transactions correctly identified as genuine.")
    print(f"FP (False Positive) : {fp} → Genuine transactions incorrectly flagged as fraud.")
    print(f"FN (False Negative) : {fn} → Fraud transactions missed by the model.")
    print(f"TP (True Positive)  : {tp} → Fraud transactions correctly detected by the model.")

    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\n" + "-" * 70)
    print("WHAT DO THESE RESULTS MEAN?")
    print("-" * 70)
    print(f"Accuracy Reason  : The model correctly classified {correctly_classified} out of {total_transactions} total transactions.")
    print(f"Precision Reason : Out of {predicted_fraud} transactions predicted as fraud, {tp} were actually fraud.")
    print(f"Recall Reason    : Out of {actual_fraud} actual fraud transactions, the model detected {tp} and missed {fn}.")
    print("F1 Score Reason  : F1-score combines precision and recall into a single balance measure.")
    print(f"ROC-AUC Reason   : A score of {roc_auc:.4f} indicates strong separation between genuine and fraud behaviour.")

    print("\n" + "-" * 70)
    print("FRAUD DETECTION INTERPRETATION")
    print("-" * 70)
    if recall >= 0.90:
        print(f"✅ Recall Observation: High fraud capture rate — {tp} of {actual_fraud} fraud transactions were detected ({recall * 100:.2f}%).")
    else:
        print("⚠️ Recall Observation: Some actual fraud transactions were not detected.")
    print("⚠️ Precision Observation: Some genuine transactions were also flagged for step-up biometric verification.")

    print(f"\nActual fraud transactions : {actual_fraud}")
    print(f"Correctly detected fraud  : {tp}")
    print(f"Missed fraud              : {fn}")
    print(f"False fraud alerts        : {fp}")

    print("\n" + "-" * 70)
    print("RECONSTRUCTION ERROR ANALYSIS")
    print("-" * 70)
    print(f"Genuine mean   : {genuine_mean:.6f}")
    print(f"Fraud mean     : {fraud_mean:.6f}")
    print(f"Genuine median : {genuine_median:.6f}")
    print(f"Fraud median   : {fraud_median:.6f}")

    print(f"\nThreshold      : {threshold:.6f}")
    print("Decision rule  : Reconstruction Error > Threshold → Anomaly/Fraud Flag")

    print("\n" + "-" * 70)
    print("DETECTION SUMMARY")
    print("-" * 70)
    print(f"Fraud detected : {predicted_fraud} (True Fraud Caught: {tp})")
    print(f"Fraud missed   : {fn}")
    print(f"False alerts   : {fp}")
    print("=" * 70)

    results = {
        "threshold": threshold,
        "features_count": int(X_processed.shape[1]),
        "total_transactions": total_transactions,
        "actual_genuine": actual_genuine,
        "actual_fraud": actual_fraud,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "fraud_detected": predicted_fraud,
        "fraud_missed": fn,
        "false_alerts": fp,
        "genuine_mean_error": genuine_mean,
        "fraud_mean_error": fraud_mean,
        "genuine_median_error": genuine_median,
        "fraud_median_error": fraud_median,
    }

    joblib.dump(results, "evaluation_results.joblib")
    print("\n💾 Saved evaluation_results.joblib")
    print("\n🎉 Evaluation completed successfully.")


if __name__ == "__main__":
    evaluate_model()