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

        # --- XAI LOGIC (Dynamic & Error-Free) ---
        # 1. Feature-wise absolute error nikalna
        feature_errors = np.abs(features - reconstruction)[0]

        # 2. Dynamic feature names generation (Jitne columns honge, utne automatic names ban jayenge)
        feature_names = [f"Feature_{i+1}" for i in range(features.shape[1])]

        # 3. Sabse zyada error wale top 3 features nikalna
        top_indices = np.argsort(feature_errors)[::-1][:3]
        top_reasons = [f"{feature_names[i]} anomaly detected" for i in top_indices]
        xai_explanation = ", ".join(top_reasons)
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