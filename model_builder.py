from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim):
    """
    Builds and compiles an Autoencoder model.
    input_dim: Number of features in our dataset (e.g., 19)
    """
    # 1. ENCODER (Compressing the data)
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(14, activation="relu")(input_layer)
    encoder = Dense(7, activation="relu")(encoder) # The "Bottleneck"

    # 2. DECODER (Reconstructing the data)
    decoder = Dense(14, activation="relu")(encoder)
    output_layer = Dense(input_dim, activation="linear")(decoder)

    # 3. Combine into one model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # 4. Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

if __name__ == "__main__":
    # Testing if the model builds successfully (assuming 19 features from Phase 1)
    model = build_autoencoder(input_dim=19)
    print(">>> Autoencoder Model Built Successfully! <<<")
    model.summary()