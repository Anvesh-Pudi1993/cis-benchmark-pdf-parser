import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import hashlib
import json
import datetime

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

# Step 3: Provenance Metadata Generation
def generate_provenance_metadata(model, data):
    metadata = {
        "model_name": "CNN_MNIST",
        "model_creation_date": str(datetime.datetime.now()),
        "data_hash": hashlib.sha256(data[0].tobytes()).hexdigest(),
        "model_layers": [layer.name for layer in model.layers],
        "dependencies": ["tensorflow", "keras", "numpy"],
    }
    return metadata

# Step 4: OCTAVE Allegro Risk Assessment
def octave_risk_assessment(metadata):
    risk_factors = {
        "data_integrity": "High" if len(metadata["data_hash"]) != 64 else "Low",
        "dependency_integrity": "Low" if "tensorflow" in metadata["dependencies"] else "Critical",
        "model_architecture": "Low" if len(metadata["model_layers"]) > 1 else "Critical",
    }
    return risk_factors

# Step 5: RSA Archer Integration Simulation
def rsa_archer_report(metadata, risks):
    report = {
        "metadata": metadata,
        "risks": risks,
        "recommendations": {
            "data_integrity": "Validate dataset source and integrity using checksum.",
            "dependency_integrity": "Use verified package managers and check library versions.",
            "model_architecture": "Ensure architecture follows organizational guidelines.",
        },
    }
    return json.dumps(report, indent=4)

# Step 6: Train and evaluate the model with provenance check
def main():
    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Create CNN
    model = create_cnn()

    # Train the model
    model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

    # Generate provenance metadata
    metadata = generate_provenance_metadata(model, (x_train, y_train))

    # Perform risk assessment using OCTAVE Allegro principles
    risks = octave_risk_assessment(metadata)

    # Simulate RSA Archer reporting
    report = rsa_archer_report(metadata, risks)

    # Print the provenance report
    print("Model Provenance Report:")
    print(report)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_acc}")

if __name__ == "__main__":
    main()
