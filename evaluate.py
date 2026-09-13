
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model():

    print("🔍 [1/4] Loading trained Autoencoder model...")
    autoencoder = load_model("zero_day_autoencoder.keras")

    print("📊 [2/4] Loading test data...")
    df = pd.read_csv("data/transactions_test.csv")

    # Feature Selection
    columns_to_drop = ['transaction_id', 'user_name']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Encoding
    encoder = LabelEncoder()
    categorical_cols = df.select_dtypes(
        include=['object', 'string', 'category']
    ).columns

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Separate features and target
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    print("✅ Test data preprocessed.")
    print(f"Test transactions: {len(X_scaled)}")
    print("🚫 SMOTE NOT applied to test data.")

    # Prediction
    print("⚙️ [3/4] Calculating reconstruction errors...")
    X_array = X_scaled.to_numpy()
    y_array = y.to_numpy()

    X_pred = autoencoder.predict(X_array, verbose=0)

    # Reconstruction Error
    mse_errors = np.mean(
        np.power(X_array - X_pred, 2),
        axis=1
    )

    # Threshold
    threshold = 0.9853

    # Prediction
    y_pred = (mse_errors > threshold).astype(int)

    # Metrics
    cm = confusion_matrix(y_array, y_pred)

    accuracy = accuracy_score(y_array, y_pred)
    precision = precision_score(y_array, y_pred, zero_division=0)
    recall = recall_score(y_array, y_pred, zero_division=0)
    f1 = f1_score(y_array, y_pred, zero_division=0)

    print("\n" + "=" * 45)
    print("📈 --- ZERO-DAY FRAUD DETECTION RESULTS ---")
    print("=" * 45)

    print(f"Total Transactions: {len(y_array)}")
    print(f"Threshold: {threshold}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nPerformance Metrics:")
    print(f"Accuracy :  {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall   :  {recall:.4f}")
    print(f"F1 Score :  {f1:.4f}")

    print("\nAverage Reconstruction Error:")
    print(f"Genuine (0): {mse_errors[y_array == 0].mean():.4f}")
    print(f"Fraud   (1): {mse_errors[y_array == 1].mean():.4f}")

    print("=" * 45)


if __name__ == "__main__":
    evaluate_model()
