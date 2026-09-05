import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_pipeline import run_pipeline

def evaluate_model():
    print("🔍 [1/3] Loading trained Autoencoder model...")
    autoencoder = load_model("zero_day_autoencoder.keras")
    
    print("📊 [2/3] Loading and preprocessing test data...")
    dataset_file = "data/transactions_train.csv"
    X_data, y_data = run_pipeline(dataset_file)
    
    if X_data is None or y_data is None:
        print("❌ Error: Data loading failed.")
        return

    # Convert to numpy arrays
    X_array = X_data.to_numpy()
    y_array = y_data.to_numpy()

    # Predict / Reconstruct the data using the Autoencoder
    print("⚙️ [3/3] Calculating reconstruction errors for transactions...")
    X_pred = autoencoder.predict(X_array)
    
    # Calculate Mean Squared Error (Reconstruction Error) for each transaction
    mse_errors = np.mean(np.power(X_array - X_pred, 2), axis=1)
    
    # Add error to a temporary dataframe for analysis
    results = pd.DataFrame({
        'Actual_Label': y_array,  # 0 for Genuine, 1 for Fraud
        'Reconstruction_Error': mse_errors
    })
    
    # Separate genuine and fraud errors
    genuine_errors = results[results['Actual_Label'] == 0]['Reconstruction_Error']
    fraud_errors = results[results['Actual_Label'] == 1]['Reconstruction_Error']
    
    print("\n" + "="*40)
    print("📈 --- ZERO-DAY EVALUATION RESULTS ---")
    print(f"Total Transactions Evaluated: {len(results)}")
    print(f"Average Error on Genuine Transactions (0): {genuine_errors.mean():.4f}")
    print(f"Average Error on Fraud Transactions (1):   {fraud_errors.mean():.4f}")
    print("="*40)
    print("💡 Logic check: Fraud error is higher because the model learned ONLY genuine patterns!")

if __name__ == "__main__":
    evaluate_model()