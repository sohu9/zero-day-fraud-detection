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

    print("\n" + "=" * 70)
    print("📈 ZERO-DAY FRAUD DETECTION — FUTURE TEST RESULTS")
    print("=" * 70)
    print(f"Test Samples         : {len(y)}")
    print(f"Genuine Transactions : {int((y == 0).sum())}")
    print(f"Fraud Transactions   : {int((y == 1).sum())}")
    print(f"Accuracy             : {accuracy * 100:.4f}%")
    print(f"Precision            : {precision * 100:.4f}%")
    print(f"Recall               : {recall * 100:.4f}%")
    print(f"F1-Score             : {f1 * 100:.4f}%")
    print(f"ROC-AUC              : {roc_auc:.4f}")
    print(f"Threshold            : {threshold:.4f}")
    print(f"True Positive (TP)   : {tp}")
    print(f"True Negative (TN)   : {tn}")
    print(f"False Positive (FP)  : {fp}")
    print(f"False Negative (FN)  : {fn}")
    print("=" * 70)

    genuine_errors = mse_errors[y == 0]
    fraud_errors = mse_errors[y == 1]

    results = {
        "threshold": threshold,
        "features_count": int(X_processed.shape[1]),
        "total_transactions": int(len(y)),
        "actual_genuine": int((y == 0).sum()),
        "actual_fraud": int((y == 1).sum()),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "fraud_detected": int((y_pred == 1).sum()),
        "fraud_missed": fn,
        "false_alerts": fp,
        "genuine_mean_error": float(genuine_errors.mean()),
        "fraud_mean_error": float(fraud_errors.mean()),
        "genuine_median_error": float(np.median(genuine_errors)),
        "fraud_median_error": float(np.median(fraud_errors)),
    }

    joblib.dump(results, "evaluation_results.joblib")
    print("\n💾 Saved evaluation_results.joblib")
    print("🎉 Evaluation completed successfully.")


if __name__ == "__main__":
    evaluate_model()