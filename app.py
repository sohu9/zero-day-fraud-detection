from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained Autoencoder model into memory
print("Loading Model...")
model = tf.keras.models.load_model('zero_day_autoencoder.keras')

# Set the anomaly detection threshold derived from evaluation results
THRESHOLD = 0.9853 

@app.route('/', methods=['GET'])
def home():
    # Base endpoint to check if the API is running
    return jsonify({"status": "Zero-Day Fraud Detection API is Online!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming JSON request data from the frontend
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        # Reconstruct the input features using the trained model
        reconstruction = model.predict(features)
        
        # Calculate the Mean Squared Error (MSE) between original and reconstructed data
        mse = np.mean(np.power(features - reconstruction, 2), axis=1)[0]
        
        # Determine the transaction status based on the threshold
        if mse > THRESHOLD:
            status = "Hold & Verify 🚨"
            risk = "High"
        else:
            status = "Approved ✅"
            risk = "Low"
            
        # Return the final decision as a JSON response
        return jsonify({
            "transaction_status": status,
            "risk_level": risk,
            "reconstruction_error": float(mse)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run the Flask server on default port 5000
    app.run(port=5000)
