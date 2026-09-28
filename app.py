import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from database import (
    init_database,
    save_transaction,
    get_transactions,
    get_transaction
)

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from flask_cors import CORS

from data_pipeline import add_time_features, ID_COLUMNS, TARGET_COLUMN


app = Flask(__name__)
init_database()
CORS(app)

# WebSockets initialize
socketio = SocketIO(app, cors_allowed_origins="*")

LOG_FILE = "logs/transactions.log"
os.makedirs("logs", exist_ok=True)

MODEL_FILE = "zero_day_autoencoder.keras"
PREPROCESSOR_FILE = "preprocessor.joblib"
THRESHOLD_FILE = "threshold.joblib"
TEST_FILE = "data/transactions_test.csv"


# ============================================================
# LOAD TRAINED MODEL + TRAINING PREPROCESSOR + THRESHOLD
# ============================================================

print("Loading Model...")
model = tf.keras.models.load_model(MODEL_FILE)
print("✅ Model loaded.")

print("Loading training preprocessing artifact...")
preprocessor_artifact = joblib.load(PREPROCESSOR_FILE)

preprocessor = preprocessor_artifact["preprocessor"]
feature_names = preprocessor_artifact["feature_names"]

print("✅ Preprocessor loaded.")
print(f"Number of features: {len(feature_names)}")

print("Loading calibrated threshold...")
threshold_data = joblib.load(THRESHOLD_FILE)
THRESHOLD = float(threshold_data["threshold"])

print(f"✅ Threshold loaded: {THRESHOLD:.6f}")


# ============================================================
# LOGGING
# ============================================================

def log_transaction(data, status, risk, mse):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"{datetime.now()} | "
            f"Transaction ID: {data.get('transaction_id')} | "
            f"Customer ID: {data.get('customer_id')} | "
            f"Merchant ID: {data.get('merchant_id')} | "
            f"Amount: {data.get('transaction_amount')} | "
            f"Status: {status} | "
            f"Risk: {risk} | "
            f"Reconstruction Error: {mse:.6f}\n"
        )


# ============================================================
# PREPROCESS TRANSACTION USING TRAINING PREPROCESSOR
# ============================================================

def preprocess_data(data):
    """
    Apply exactly the same preprocessing used during training.
    IDs are preserved in the transaction record but excluded
    from ML features.
    """

    df = pd.DataFrame([data])

    # Remove target if accidentally sent
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    # Remove IDs from ML input
    df = df.drop(
        columns=[col for col in ID_COLUMNS if col in df.columns],
        errors="ignore"
    )

    # Convert transaction_time into the same time features
    df = add_time_features(df)

    # Ensure same feature columns as training
    expected_columns = (
        preprocessor_artifact["numerical_columns"]
        + preprocessor_artifact["categorical_columns"]
    )

    missing_columns = [
        col for col in expected_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required transaction fields: {missing_columns}"
        )

    df = df[expected_columns]

    # Apply SAVED training preprocessor
    processed = preprocessor.transform(df)

    return np.asarray(processed, dtype=np.float32)


# ============================================================
# HUMAN-READABLE XAI
# ============================================================

def generate_xai_explanation(data, features, reconstruction):
    """
    Generate explanations from the actual dataset features.
    No fake Feature_1 / Feature_2 mapping.
    """

    feature_errors = np.abs(features - reconstruction)[0]

    top_indices = np.argsort(feature_errors)[-3:][::-1]

    explanations = []

    for index in top_indices:

        feature = feature_names[index]
        error = float(feature_errors[index])

        # ----------------------------------------------------
        # Transaction Amount
        # ----------------------------------------------------
        if feature == "transaction_amount":
            amount = data.get("transaction_amount")

            if amount is not None:
                explanations.append(
                    f"Transaction amount ₹{float(amount):,.2f} "
                    f"shows an unusual pattern"
                )
            else:
                explanations.append(
                    "Transaction amount shows an unusual pattern"
                )

        # ----------------------------------------------------
        # Amount deviation
        # ----------------------------------------------------
        elif feature == "amount_deviation_from_user_mean":
            value = data.get("amount_deviation_from_user_mean")

            if value is not None:
                explanations.append(
                    f"Transaction amount deviation from the user's "
                    f"normal spending pattern is high ({float(value):.2f})"
                )
            else:
                explanations.append(
                    "Transaction amount deviation from normal spending "
                    "pattern is unusually high"
                )

        # ----------------------------------------------------
        # Transaction frequency
        # ----------------------------------------------------
        elif feature == "txn_count_1h":
            value = data.get("txn_count_1h")

            if value is not None:
                explanations.append(
                    f"High transaction activity detected in the last hour "
                    f"({int(float(value))} transactions)"
                )
            else:
                explanations.append(
                    "Unusual transaction activity detected within one hour"
                )

        elif feature == "txn_count_24h":
            value = data.get("txn_count_24h")

            if value is not None:
                explanations.append(
                    f"Unusual transaction activity in the last 24 hours "
                    f"({int(float(value))} transactions)"
                )
            else:
                explanations.append(
                    "Unusual transaction activity detected within 24 hours"
                )

        # ----------------------------------------------------
        # Failed transactions
        # ----------------------------------------------------
        elif feature == "failed_txn_count_24h":
            value = data.get("failed_txn_count_24h")

            if value is not None:
                explanations.append(
                    f"Elevated failed transaction count in the last 24 hours "
                    f"({int(float(value))} failures)"
                )
            else:
                explanations.append(
                    "Elevated failed transaction activity detected"
                )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------
        elif feature == "device_type":
            device = data.get("device_type", "unknown")

            explanations.append(
                f"Device type '{device}' contributes strongly to the anomaly score"
            )

        # ----------------------------------------------------
        # Payment channel
        # ----------------------------------------------------
        elif feature == "payment_channel":
            channel = data.get("payment_channel", "unknown")

            explanations.append(
                f"Payment channel '{channel}' contributes strongly to the anomaly score"
            )

        # ----------------------------------------------------
        # IP risk
        # ----------------------------------------------------
        elif feature == "ip_risk_score":
            value = data.get("ip_risk_score")

            if value is not None:
                explanations.append(
                    f"Elevated IP risk score detected ({float(value):.2f})"
                )
            else:
                explanations.append(
                    "IP-related risk contributes strongly to the anomaly score"
                )

        # ----------------------------------------------------
        # Merchant risk
        # ----------------------------------------------------
        elif feature == "merchant_risk_score":
            value = data.get("merchant_risk_score")

            if value is not None:
                explanations.append(
                    f"Merchant risk score is elevated ({float(value):.2f})"
                )
            else:
                explanations.append(
                    "Merchant risk contributes strongly to the anomaly score"
                )

        # ----------------------------------------------------
        # Geographic distance
        # ----------------------------------------------------
        elif feature == "geo_distance_from_last_txn":
            value = data.get("geo_distance_from_last_txn")

            if value is not None:
                explanations.append(
                    f"Large geographic distance from the previous transaction "
                    f"({float(value):.2f})"
                )
            else:
                explanations.append(
                    "Geographic transaction pattern is unusual"
                )

        # ----------------------------------------------------
        # International transaction
        # ----------------------------------------------------
        elif feature == "is_international":
            value = data.get("is_international")

            if value is not None and int(float(value)) == 1:
                explanations.append(
                    "International transaction contributes to the anomaly score"
                )
            else:
                explanations.append(
                    "International transaction behavior contributes to the anomaly score"
                )

        # ----------------------------------------------------
        # Post-auth risk
        # ----------------------------------------------------
        elif feature == "post_auth_risk_score":
            value = data.get("post_auth_risk_score")

            if value is not None:
                explanations.append(
                    f"Post-authentication risk score is elevated "
                    f"({float(value):.2f})"
                )
            else:
                explanations.append(
                    "Post-authentication risk contributes strongly to the anomaly score"
                )

        # ----------------------------------------------------
        # Account age
        # ----------------------------------------------------
        elif feature == "account_age_days":
            explanations.append(
                "Account age contributes strongly to the detected anomaly"
            )

        # ----------------------------------------------------
        # Credit / KYC
        # ----------------------------------------------------
        elif feature == "credit_score_band":
            explanations.append(
                "Credit-score band contributes strongly to the detected anomaly"
            )

        elif feature == "kyc_level":
            explanations.append(
                "KYC level contributes strongly to the detected anomaly"
            )

        # ----------------------------------------------------
        # Time features
        # ----------------------------------------------------
        elif feature == "transaction_hour":
            explanations.append(
                "Transaction timing contributes strongly to the anomaly score"
            )

        elif feature == "transaction_day_of_week":
            explanations.append(
                "Transaction day pattern contributes strongly to the anomaly score"
            )

        elif feature == "is_weekend":
            explanations.append(
                "Weekend transaction pattern contributes strongly to the anomaly score"
            )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------
        else:
            explanations.append(
                f"Feature '{feature}' contributes strongly to the anomaly score"
            )

    return " | ".join(explanations)


# ============================================================
# PREDICT ONE TRANSACTION
# ============================================================

def analyze_transaction(data):

    features = preprocess_data(data)

    reconstruction = model.predict(
        features,
        verbose=0
    )

    mse = float(
        np.mean(
            np.square(features - reconstruction),
            axis=1
        )[0]
    )

    if mse > THRESHOLD:

        status = "Hold & Verify 🚨"
        risk = "High"

        xai_explanation = generate_xai_explanation(
            data,
            features,
            reconstruction
        )

    else:

        status = "Approved ✅"
        risk = "Low"

        xai_explanation = (
            "Transaction reconstruction error is below "
            "the anomaly threshold."
        )

    return {
        "transaction_id": data.get("transaction_id"),
        "customer_id": data.get("customer_id"),
        "merchant_id": data.get("merchant_id"),
        "transaction_amount": data.get("transaction_amount"),
        "transaction_status": status,
        "risk_level": risk,
        "reconstruction_error": mse,
        "threshold": THRESHOLD,
        "xai_explanation": xai_explanation
    }


# ============================================================
# REAL-TIME AI TRANSACTION STREAM
# ============================================================

stream_started = False


def background_transaction_stream():

    print("Starting AI Transaction Stream... 🚀")

    try:

        df = pd.read_csv(TEST_FILE)

        for index, row in df.iterrows():

            socketio.sleep(3)

            data = row.to_dict()

            try:

                result = analyze_transaction(data)
                save_transaction(result)

                log_transaction(
                    data,
                    result["transaction_status"],
                    result["risk_level"],
                    result["reconstruction_error"]
                )

                amount = data.get(
                    "transaction_amount",
                    0
                )

                merchant = data.get(
                    "merchant_id",
                    "Unknown"
                )

                socketio.emit(
                    "new_transaction",
                    {
                        **result,
                        "message":
                            f"Merchant: {merchant} | "
                            f"Amount: ₹{amount} | "
                            f"Status: {result['transaction_status']}"
                    }
                )

            except Exception as e:

                print(
                    f"Row {index} skipped because of error: {e}"
                )

    except Exception as e:

        print(
            "CSV file read error:",
            e
        )


# ============================================================
# SOCKET CONNECTION
# ============================================================

@socketio.on("connect")
def handle_connect():

    global stream_started

    print(
        "Admin Dashboard Connected to Live Stream! 🟢"
    )

    # Prevent multiple streams when browser reconnects
    if not stream_started:

        stream_started = True

        socketio.start_background_task(
            background_transaction_stream
        )


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# MANUAL PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No transaction data received."
            }), 400

        result = analyze_transaction(data)
        save_transaction(result)

        log_transaction(
            data,
            result["transaction_status"],
            result["risk_level"],
            result["reconstruction_error"]
        )

        return jsonify(result)

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 400
# ============================================================
# TRANSACTION HISTORY APIs
# ============================================================

@app.route("/api/transactions", methods=["GET"])
def api_transactions():

    filter_type = request.args.get("type", "all")

    transactions = get_transactions(filter_type)

    return jsonify({
        "count": len(transactions),
        "transactions": transactions
    })


@app.route("/api/transactions/<transaction_id>", methods=["GET"])
def api_transaction_detail(transaction_id):

    transaction = get_transaction(transaction_id)

    if transaction is None:
        return jsonify({
            "error": "Transaction not found"
        }), 404

    return jsonify(transaction)
    


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("🚀 ZERO-DAY FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print(f"Model     : {MODEL_FILE}")
    print(f"Features  : {len(feature_names)}")
    print(f"Threshold : {THRESHOLD:.6f}")
    print("=" * 60)

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        allow_unsafe_werkzeug=True
    )