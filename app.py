from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from preprocess_live import prepare_preprocessor, preprocess_transaction

app = Flask(__name__)
LOG_FILE = "logs/transactions.log"

os.makedirs("logs", exist_ok=True)


def log_transaction(data, status, risk, mse):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"{datetime.now()} | "
            f"Customer ID: {data.get('customer_id')} | "
            f"Merchant ID: {data.get('merchant_id')} | "
            f"Amount: {data.get('transaction_amount')} | "
            f"Status: {status} | "
            f"Risk: {risk} | "
            f"Reconstruction Error: {mse:.6f}\n"
        )
# Load the trained Autoencoder model into memory
print("Loading Model...")
model = tf.keras.models.load_model("zero_day_autoencoder.keras")

# Prepare preprocessing using the training dataset
print("Preparing preprocessing...")
encoders, scaler, feature_columns = prepare_preprocessor()
print("Preprocessing ready.")
print(f"Number of features: {len(feature_columns)}")

# Anomaly detection threshold
THRESHOLD = 0.9853


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive transaction data from frontend
        data = request.json

        # Convert frontend data into the same format
        # used during model training
        features = preprocess_transaction(
            data,
            encoders,
            scaler,
            feature_columns
        )

        # Reconstruct the transaction using Autoencoder
        reconstruction = model.predict(features, verbose=0)

        # Calculate reconstruction error
        mse = np.mean(
            np.power(features - reconstruction, 2),
            axis=1
        )[0]

        # Compare reconstruction error with threshold
        if mse > THRESHOLD:
            status = "Hold & Verify 🚨"
            risk = "High"
        else:
            status = "Approved ✅"
            risk = "Low"
        log_transaction(data, status, risk, mse)
        # Return result to frontend
        return jsonify({
            "transaction_status": status,
            "risk_level": risk,
            "reconstruction_error": float(mse),
            "threshold": THRESHOLD
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(port=5000)