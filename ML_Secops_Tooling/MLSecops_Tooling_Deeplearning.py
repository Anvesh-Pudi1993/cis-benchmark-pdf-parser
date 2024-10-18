import os
import pickle
import hashlib
import numpy as np
import pandas as pd
import logging
import psutil
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from cryptography.fernet import Fernet
from sklearn.ensemble import IsolationForest
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method  # Ensure correct import

# Set up logging
logging.basicConfig(filename='dev_catalog_threat_check_deeplearning.log', level=logging.INFO)

# 1. Load CIFAR-10 dataset
def load_cifar10_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize images
    return X_train, y_train, X_test, y_test

# 2. Define and train a simple CNN model (deep learning)
def train_model(X_train, y_train):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation=None)  # Use logits here, not softmax for adversarial attacks
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_split=0.1)
    return model

# 3. Serialize and encrypt the model
def serialize_and_encrypt_model(model, encryption_key):
    fernet = Fernet(encryption_key)
    serialized_model = model.to_json()  # Serialize model structure
    encrypted_model = fernet.encrypt(serialized_model.encode())  # Encrypt model structure
    return encrypted_model

# 4. Save encrypted model to local storage
def save_to_local(file_path, encrypted_data):
    with open(file_path, 'wb') as file:
        file.write(encrypted_data)
    logging.info(f"Model saved and encrypted at {file_path}")

# 5. Check model integrity (hash comparison)
def verify_model_integrity(file_path, expected_hash):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        current_hash = sha256_hash.hexdigest()
        if current_hash == expected_hash:
            logging.info(f"Model integrity verified for {file_path}")
        else:
            logging.warning(f"Model integrity check failed for {file_path}")
    except Exception as e:
        logging.error(f"Error verifying model integrity: {e}")

# 6. Check for data poisoning (outlier detection)
def check_data_poisoning(X):
    clf = IsolationForest(contamination=0.01)
    clf.fit(X.reshape(X.shape[0], -1))  # Reshape for outlier detection
    outliers = clf.predict(X.reshape(X.shape[0], -1))
    num_outliers = (outliers == -1).sum()
    if num_outliers > 0:
        logging.warning(f"Data Poisoning Alert: {num_outliers} outliers detected in feature set")
    else:
        logging.info("No significant outliers detected (No data poisoning)")

# 7. Monitor system resources to detect potential DoS attacks
def monitor_system_resources():
    cpu_usage = psutil.cpu_percent()
    if cpu_usage > 80:  # Threshold can be adjusted
        logging.warning(f"High CPU usage detected: {cpu_usage}%")
    else:
        logging.info(f"CPU usage is normal: {cpu_usage}%")

# 8. Check model storage access (Unauthorized Access)
def check_storage_access(storage_path):
    if not os.path.exists(storage_path):
        logging.warning(f"Storage path does not exist: {storage_path}")
    else:
        logging.info(f"Storage path exists: {storage_path}")

# 9. Check model robustness against adversarial attacks
def check_adversarial_attack(model, X_test, y_test):
    # Generate adversarial examples using FGSM
    epsilon = 0.1  # Perturbation magnitude
    X_test_adv = fast_gradient_method(model, X_test, eps=epsilon, norm=np.inf)

    # Evaluate model on adversarial examples
    logits = model(X_test_adv)
    predicted_classes = np.argmax(logits, axis=1)

    # Calculate accuracy on adversarial examples
    adversarial_accuracy = np.mean(predicted_classes == y_test.flatten())
    if adversarial_accuracy < 0.5:  # Threshold can be adjusted
        logging.warning(f"Adversarial Attack Detected: Accuracy on adversarial examples is {adversarial_accuracy:.2f}")
    else:
        logging.info(f"Model is robust against adversarial attacks: Accuracy is {adversarial_accuracy:.2f}")

# Main function to run all checks
def run_threat_vector_checks():
    # Load and train model
    X_train, y_train, X_test, y_test = load_cifar10_data()
    model = train_model(X_train, y_train)

    # Encrypt model
    encryption_key = Fernet.generate_key()
    encrypted_model = serialize_and_encrypt_model(model, encryption_key)
    save_to_local("encrypted_model.json", encrypted_model)

    # Hash model to verify integrity
    model_hash = hashlib.sha256(model.to_json().encode()).hexdigest()
    verify_model_integrity("encrypted_model.json", model_hash)

    # Check for data poisoning
    check_data_poisoning(X_train)

    # Monitor system resources
    monitor_system_resources()

    # Check storage access
    check_storage_access("encrypted_model.json")

    # Check for adversarial attacks
    check_adversarial_attack(model, X_test, y_test)

# Run all threat vector checks
if __name__ == "__main__":
    run_threat_vector_checks()
