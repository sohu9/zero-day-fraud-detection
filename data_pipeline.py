import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def run_pipeline(data_path):
    print("🚀 1. Data Pipeline Started...")
    
    # Load Data
    try:
        df = pd.read_csv(data_path)
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Dataset file not found. Check the file path.")
        return None, None

    # ---------------------------------------------------------
    # STEP 1: Feature Selection (Waseem & Abdurrahman's logic)
    # ---------------------------------------------------------
    columns_to_drop = ['transaction_id', 'user_name'] 
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print("✅ 2. Useless columns dropped (Feature Selection).")

   # ---------------------------------------------------------
    # STEP 2: Label Encoding (Aun)
    # ---------------------------------------------------------
    encoder = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
    print("✅ 3. Categorical text converted to numbers (Encoding).")
    # Assuming 'is_fraud' is our target column (0 for genuine, 1 for fraud)
    if 'is_fraud' in df.columns:
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']
    else:
        X = df
        y = None

    # ---------------------------------------------------------
    # STEP 3: Feature Scaling (Waseem & Abdurrahman's logic)
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print("✅ 4. All features balanced properly (Scaling).")

    # ---------------------------------------------------------
    # STEP 4: SMOTE Balancing (Aun's logic)
    # ---------------------------------------------------------
    if y is not None:
        smote = SMOTE(random_state=42)
        X_final, y_final = smote.fit_resample(X_scaled, y)
        print("✅ 5. Fraud cases artificially balanced (SMOTE).")
    else:
        X_final, y_final = X_scaled, None

    print("🎉 Pipeline Complete! Data is ready for the Autoencoder Brain.")
    return X_final, y_final

# Testing the pipeline
if __name__ == "__main__":
    # Yahan path me 'data/' lagana zaroori hai kyunki file us folder ke andar hai
    dataset_file = "data/transactions_train.csv" 
    
    print(f"Loading dataset from: {dataset_file}")
    X_train, y_train = run_pipeline(dataset_file)
    
    if X_train is not None:
        print(f"✅ Final Data Shape: {X_train.shape}")