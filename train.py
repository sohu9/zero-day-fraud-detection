from data_pipeline import run_pipeline
from model_builder import build_autoencoder
import numpy as np

def train_fraud_detector():
    print("🚀 Starting Zero-Day Fraud Detection Training...")
    
    # 1. Load and Preprocess Data via Pipeline
    dataset_file = "data/transactions_train.csv"
    X_train, y_train = run_pipeline(dataset_file)
    
    if X_train is None:
        print("❌ Training stopped due to data loading error.")
        return

    # 2. IMPORTANT FOR AUTOENCODER: 
    # Autoencoder ko hum sirf "Genuine" transactions (y == 0) par train karenge,
    # taaki wo sirf normal patterns seekhe aur fraud aane par reconstruct na kar paye.
    if y_train is not None:
        # Convert to numpy and filter only non-fraud (0) rows
        X_train_array = X_train.to_numpy()
        y_train_array = y_train.to_numpy()
        
        X_genuine = X_train_array[y_train_array == 0]
        print(f"🔒 Training Autoencoder ONLY on Genuine transactions. Shape: {X_genuine.shape}")
    else:
        X_genuine = X_train.to_numpy()

    # 3. Build Model (Input dimension = number of columns/features)
    input_dim = X_genuine.shape[1]
    print(f"🧠 Building Autoencoder for {input_dim} features...")
    autoencoder = build_autoencoder(input_dim)

    # 4. Train the Model
    print("🏋️‍♂️ Training model in progress...")
    history = autoencoder.fit(
        X_genuine, X_genuine,  # Autoencoder inputs target ke barabar hi hota hai (reconstruction)
        epochs=20,             # Iterations
        batch_size=32,
        shuffle=True,
        validation_split=0.1   # 10% data validation ke liye
    )

    # 5. Save the trained model weights
    autoencoder.save("zero_day_autoencoder.keras")
    print("🎉 Training Complete! Model saved successfully as 'zero_day_autoencoder.keras'.")

if __name__ == "__main__":
    train_fraud_detector()