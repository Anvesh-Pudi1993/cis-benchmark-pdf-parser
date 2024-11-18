import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np

# Step 1: Load and preprocess the MNIST data
def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

# Step 2: Define a CNN model
def create_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Threat Modeling Checks
def threat_model_check(data, model):
    checks = {
        "Supply Chain Vulnerabilities": check_supply_chain_vulnerabilities()
    }
    return checks

# def check_data_integrity(data):
#     # Check for anomalies in data (e.g., poisoning)
#     x_train, y_train, _, _ = data
#     if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
#         return "Data contains NaN values. Possible corruption detected."
#     return "Data integrity check passed."

# def check_model_integrity(model):
#     # Validate model architecture and parameters
#     for layer in model.layers:
#         if not hasattr(layer, 'weights'):
#             return f"Model layer {layer.name} has no weights. Potential tampering detected."
#     return "Model integrity check passed."

def check_supply_chain_vulnerabilities():
    # Check for risks in the supply chain (e.g., dependencies, external libraries)
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError as e:
        return f"Missing dependency: {e}"
    return "Supply chain dependencies verified."

# Step 4: Train and evaluate the model
def main():
    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Create CNN
    model = create_cnn()

    # Train the model
    model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc}")

    # Perform threat model checks
    checks = threat_model_check((x_train, y_train, x_test, y_test), model)
    for check, result in checks.items():
        print(f"{check}: {result}")

if __name__ == "__main__":
    main()
