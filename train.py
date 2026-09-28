import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from data_pipeline import run_pipeline
from model_builder import build_autoencoder


MODEL_FILE = "zero_day_autoencoder.keras"
THRESHOLD_FILE = "threshold.joblib"


def train_fraud_detector():

    print("🚀 Starting Zero-Day Fraud Detection Training...")

    # ---------------------------------------------------------
    # 1. Load and preprocess training data
    # ---------------------------------------------------------

    X, y, feature_names = run_pipeline(
        "data/transactions_train.csv",
        save_preprocessor=True
    )

    print(
        f"\nTotal training transactions: {len(X)}"
    )

    # ---------------------------------------------------------
    # 2. Keep ONLY genuine transactions
    # ---------------------------------------------------------

    y_array = y.to_numpy()

    X_genuine = X[y_array == 0]

    print(
        f"🔒 Genuine transactions for Autoencoder: "
        f"{len(X_genuine)}"
    )

    # ---------------------------------------------------------
    # 3. Train / validation split
    # ---------------------------------------------------------

    split_index = int(len(X_genuine) * 0.90)

    X_train = X_genuine[:split_index]
    X_validation = X_genuine[split_index:]

    print(
        f"Training genuine samples   : {len(X_train)}"
    )

    print(
        f"Validation genuine samples : {len(X_validation)}"
    )

    # ---------------------------------------------------------
    # 4. Build Autoencoder
    # ---------------------------------------------------------

    input_dim = X_train.shape[1]

    print(
        f"\n🧠 Building Autoencoder "
        f"for {input_dim} features..."
    )

    autoencoder = build_autoencoder(input_dim)

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------

    print("\n🏋️ Training Autoencoder...")

    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=20,
        batch_size=256,
        shuffle=True,
        validation_data=(
            X_validation,
            X_validation
        ),
        verbose=1
    )

    # ---------------------------------------------------------
    # 6. Calculate validation reconstruction errors
    # ---------------------------------------------------------

    print(
        "\n🔍 Calculating validation reconstruction errors..."
    )

    validation_predictions = autoencoder.predict(
        X_validation,
        verbose=0
    )

    validation_errors = np.mean(
        np.square(
            X_validation -
            validation_predictions
        ),
        axis=1
    )

    # ---------------------------------------------------------
    # 7. Calibrate threshold
    # ---------------------------------------------------------

    # 99th percentile of genuine validation errors.
    threshold = float(
        np.percentile(
            validation_errors,
            99
        )
    )

    print(
        f"\n🎯 Calibrated threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Validation error mean: "
        f"{validation_errors.mean():.6f}"
    )

    print(
        f"Validation error max: "
        f"{validation_errors.max():.6f}"
    )

    # ---------------------------------------------------------
    # 8. Save threshold
    # ---------------------------------------------------------

    joblib.dump(
        {
            "threshold": threshold,
            "method": "99th_percentile_genuine_validation_error"
        },
        THRESHOLD_FILE
    )

    print(
        f"💾 Threshold saved to {THRESHOLD_FILE}"
    )

    # ---------------------------------------------------------
    # 9. Save model
    # ---------------------------------------------------------

    autoencoder.save(MODEL_FILE)

    print(
        f"💾 Model saved to {MODEL_FILE}"
    )

    # ---------------------------------------------------------
    # 10. Save training graph
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Autoencoder Training and Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "training_loss_graph.png",
        dpi=150
    )

    plt.close()

    print(
        "📊 Saved training_loss_graph.png"
    )

    print("\n" + "=" * 55)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print(
        f"Model       : {MODEL_FILE}"
    )

    print(
        f"Preprocessor: preprocessor.joblib"
    )

    print(
        f"Threshold   : {threshold:.6f}"
    )

    print(
        f"Features    : {input_dim}"
    )


if __name__ == "__main__":
    train_fraud_detector()