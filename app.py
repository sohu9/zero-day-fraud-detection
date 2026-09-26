from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from preprocess_live import prepare_preprocessor, preprocess_transaction

app = Flask(__name__)
CORS(app)
print("Loading Model...")
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


@app.route('/')
def home():
    return render_template('index.html')

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
            
            # --- NAYA XAI LOGIC (Top 3 suspicious features nikalne ke liye) ---
            feature_errors = np.abs(features - reconstruction)[0]
            top_indices = np.argsort(feature_errors)[-3:][::-1]
            
            reasons = []
            for idx in top_indices:
                col_name = feature_columns[idx]
                reasons.append(f"Highly unusual pattern detected in {col_name}")
            
            xai_text = " | ".join(reasons)
            # ------------------------------------------------------------------
        else:
            status = "Approved ✅"
            risk = "Low"
            xai_text = ""

        log_transaction(data, status, risk, mse)
        
        # Return result to frontend (Ab XAI bhi sath jayega!)
        return jsonify({
            "transaction_status": status,
            "risk_level": risk,
            "reconstruction_error": float(mse),
            "threshold": THRESHOLD,
            "xai_explanation": xai_text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)