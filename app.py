from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained Autoencoder model into memory
print("Loading Model...")
model = tf.keras.models.load_model("zero_day_autoencoder.keras")

# Set the anomaly detection threshold derived from evaluation results
THRESHOLD = 0.9853


@app.route("/", methods=["GET"])
def home():
    # Base endpoint to check if the API is running
    return jsonify({"status": "Zero-Day Fraud Detection API is Online!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the incoming JSON request data from the frontend
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Reconstruct the input features using the trained model
        reconstruction = model.predict(features)

        # Calculate the Mean Squared Error (MSE) between original and reconstructed data
        mse = np.mean(np.power(features - reconstruction, 2), axis=1)[0]

       # --- XAI LOGIC (Professional Banking Explanations for 19 Features) ---
        feature_errors = np.abs(features - reconstruction)[0]

        detailed_reasons = {
            0: "Transaction Amount significantly exceeds the user's historical spending baseline.",
            1: "Geographical Distance indicates an impossible travel time from the last known location.",
            2: "Transaction initiated at an anomalous Time of Day compared to regular user habits.",
            3: "IP Address flagged for high risk (potential Tor, proxy, or known malicious subnet).",
            4: "Multiple failed authentication or OTP attempts detected prior to this transaction.",
            5: "Device signature mismatch: Transaction originated from an unrecognized or cloned device.",
            6: "Outdated or spoofed OS/Browser version detected, commonly associated with automated bot scripts.",
            7: "Account age anomaly: High-value transaction from a newly created or previously dormant account.",
            8: "Recipient account has a history of flagged transactions or low trust score.",
            9: "Suspicious transaction velocity: High frequency of transfers attempted in a very short timeframe.",
            10: "Transaction made to a high-risk Merchant Category (e.g., crypto, offshore gambling) unusual for this user.",
            11: "Mismatch detected between registered billing address and current active network location.",
            12: "Anonymous network detected: Traffic routed through commercial VPN or hidden relays.",
            13: "Device integrity compromised: Mobile app is running on a rooted, jailbroken, or emulator environment.",
            14: "High-risk payment method: Use of disposable virtual cards or newly added prepaid cards.",
            15: "Sudden cross-currency conversion or foreign transaction not matching the user's financial baseline.",
            16: "Cross-border transaction triggered strict Anti-Money Laundering (AML) geographical policies.",
            17: "Large volume transfer initiated during non-banking hours (weekend/holiday) avoiding immediate manual review.",
            18: "Sudden spike in transaction payload size or API request formatting anomaly detected."
        }

        top_indices = np.argsort(feature_errors)[::-1][:3]
        top_reasons = []
        
        for i in top_indices:
            reason = detailed_reasons.get(i, f"High statistical deviation detected in transaction parameter {i}.")
            top_reasons.append(reason)

        xai_explanation = " | ".join(top_reasons)
        # ----------------------------------------

        # Determine the transaction status based on the threshold
        if mse > THRESHOLD:
            status = "Hold & Verify 🚨"
            risk = "High"
        else:
            status = "Approved ✅"
            risk = "Low"

        # Return the final decision as a JSON response (including XAI explanation)
        return jsonify({
            "transaction_status": status,
            "risk_level": risk,
            "reconstruction_error": float(mse),
            "xai_explanation": xai_explanation,
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # Run the Flask server on default port 5000
    app.run(port=5000)