<<<<<<< HEAD
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
=======
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
import numpy as np
>>>>>>> origin/feature/real-time-engine
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from preprocess_live import prepare_preprocessor, preprocess_transaction

app = Flask(__name__)
CORS(app)
print("Loading Model...")
LOG_FILE = "logs/transactions.log"

<<<<<<< HEAD
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
=======
# WebSockets initialize
socketio = SocketIO(app, cors_allowed_origins="*")

# Dummy background task jo har 2 second mein live data bhejega
def background_transaction_stream():
    count = 0
    while True:
        socketio.sleep(2)
        count += 1
        socketio.emit('new_transaction', {'id': count, 'message': f'Live Transaction #{count} Received!'})

@socketio.on('connect')
def handle_connect():
    print("Admin Dashboard Connected to Live Stream! 🟢")
    socketio.start_background_task(background_transaction_stream)

>>>>>>> origin/feature/real-time-engine
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

<<<<<<< HEAD
        # Calculate reconstruction error
        mse = np.mean(
            np.power(features - reconstruction, 2),
            axis=1
        )[0]

      # Compare reconstruction error with threshold
=======
        # --- XAI LOGIC (Professional 19-Feature User-Friendly Mapping) ---
        # 1. Feature-wise absolute error nikalna
        feature_errors = np.abs(features - reconstruction)[0]

        # 2. All 19 Features ki Professional User-Facing Descriptions Dictionary
        feature_mapping = {
            "Feature_1": "Unusual transaction amount deviation compared to typical spending pattern",
            "Feature_2": "Abnormal transaction frequency within a short time window",
            "Feature_3": "Unexpected time interval between consecutive transactions",
            "Feature_4": "Unrecognized device identifier or browser fingerprint",
            "Feature_5": "High-risk geographic location or unfamiliar IP address detected",
            "Feature_6": "Anomalous merchant category code or risky merchant profile",
            "Feature_7": "Unusual currency conversion or cross-border payment pattern",
            "Feature_8": "Abnormal account balance depletion rate",
            "Feature_9": "Suspicious login session duration prior to transaction",
            "Feature_10": "Irregular input behavior or behavioral biometrics mismatch",
            "Feature_11": "Unusual proxy or VPN network signature detected",
            "Feature_12": "Rapid consecutive failed authentication attempts",
            "Feature_13": "Mismatch in billing and shipping address profile",
            "Feature_14": "Abnormal credit utilization ratio spike",
            "Feature_15": "Unusual transaction velocity during off-peak hours",
            "Feature_16": "Suspicious interaction pattern with payment gateway interface",
            "Feature_17": "Unverified contact details or sudden profile modification",
            "Feature_18": "Anomalous multi-account transaction linkage from same device",
            "Feature_19": "High-risk behavioral score computed by risk engine"
        }

        # 3. Sabse zyada error wale top 3 features nikalna
        top_indices = np.argsort(feature_errors)[::-1][:3]
        
        # 4. Map top indices to professional user-friendly sentences
        top_reasons = []
        for i in top_indices:
            feature_key = f"Feature_{i+1}"
            reason = feature_mapping.get(feature_key, f"Anomalous pattern detected in {feature_key}")
            top_reasons.append(reason)
            
        xai_explanation = " | ".join(top_reasons)
        # ------------------------------------------------------------------

        # Determine the transaction status based on the threshold
>>>>>>> origin/feature/real-time-engine
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


<<<<<<< HEAD
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
=======
if __name__ == "__main__":
    # Run the server using socketio.run for real-time WebSocket support
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
>>>>>>> origin/feature/real-time-engine
