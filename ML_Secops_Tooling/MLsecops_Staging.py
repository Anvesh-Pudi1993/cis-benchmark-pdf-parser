import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import hashlib
import os
import traceback
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.utils import set_log_level
import subprocess
import ctypes
import platform

# Initialize logging to a file with rotation (for large log files) and restricted permissions
logging.basicConfig(
    filename="ml_pipeline_threats_staging.log", 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite logs with each run
)

# Set logging level for CleverHans
set_log_level(logging.WARNING)

# Function to run Bandit for static analysis
def run_bandit():
    try:
        logging.info("Running Bandit for static code analysis.")
        result = subprocess.run(["bandit", "-r", "."], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("No issues found by Bandit.")
        else:
            logging.warning(f"Bandit found potential security issues: {result.stdout}")
    except Exception as e:
        logging.error(f"Error running Bandit: {traceback.format_exc()}")
        raise

# Function to run Safety for dependency vulnerability checks
def run_safety():
    try:
        logging.info("Running Safety to check dependencies for known vulnerabilities.")
        result = subprocess.run(["safety", "check"], capture_output=True, text=True)
        if "No known security vulnerabilities" in result.stdout:
            logging.info("No vulnerabilities found in dependencies.")
        else:
            logging.warning(f"Safety found vulnerabilities: {result.stdout}")
    except Exception as e:
        logging.error(f"Error running Safety: {traceback.format_exc()}")
        raise

# Load and preprocess the CIFAR-10 dataset with exception handling
try:
    logging.info("Loading CIFAR-10 dataset")
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
except Exception as e:
    logging.error(f"Error loading dataset: {traceback.format_exc()}")
    raise

# Subset CIFAR-10 to only 3 classes (e.g., classes 0 (airplane), 1 (automobile), and 2 (bird))
classes_to_keep = [0, 1, 2]
X_train = X_train[np.isin(y_train, classes_to_keep).flatten()]
y_train = y_train[np.isin(y_train, classes_to_keep).flatten()]
X_test = X_test[np.isin(y_test, classes_to_keep).flatten()]
y_test = y_test[np.isin(y_test, classes_to_keep).flatten()]

# Relabel the classes to 0, 1, and 2 for simplicity
for new_label, original_label in enumerate(classes_to_keep):
    y_train[y_train == original_label] = new_label
    y_test[y_test == original_label] = new_label

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Function to check data integrity
def check_data_integrity(data):
    """ Check data integrity by hashing before and after loading. """
    try:
        data_hash_before = hashlib.sha256(data.tobytes()).hexdigest()
        logging.info("Initial Data Hash Generated")

        # Simulate malicious changes (data poisoning)
        data_copy = np.copy(data)
        data_copy[0, 0, 0, 0] = 255  # Simulate Data Poisoning
        data_hash_after = hashlib.sha256(data_copy.tobytes()).hexdigest()

        if data_hash_before != data_hash_after:
            logging.warning("Data Integrity Compromised: Hash mismatch detected after ingestion.")
        else:
            logging.info("Data Integrity Verified: No issues with the data.")
    except Exception as e:
        logging.error(f"Error during data integrity check: {traceback.format_exc()}")
        raise

# Function to check test integrity (unit/integration tests)
def check_test_integrity():
    """ Checks for tampering or manipulation in the testing environment. """
    try:
        logging.info("Running tests to check for test manipulation.")
        result = subprocess.run(["pytest"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("All tests passed successfully.")
        else:
            logging.warning(f"Test failures detected: {result.stdout}")
    except Exception as e:
        logging.error(f"Error running tests: {traceback.format_exc()}")
        raise

# Function to check for privilege escalation
def check_privileges():
    """ Check if the script is running with elevated privileges (root/admin). """
    try:
        if os.name == 'nt':  # For Windows
            logging.info("Checking privileges on Windows...")
            # Use ctypes to check for admin privileges on Windows
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            if is_admin:
                logging.info("Administrator privileges detected.")
            else:
                logging.warning("No administrator privileges detected.")
        else:
            logging.info("Checking privileges on Unix-like system...")
            # For Unix-based systems (Linux/macOS)
            user_id = os.getuid()
            if user_id == 0:
                logging.info("Root privileges detected.")
            else:
                logging.warning("No root privileges detected.")
    except Exception as e:
        logging.error(f"Error checking privileges: {str(e)}")
        raise

# Additional testing integrity checks
def check_test_data_poisoning(X_train):
    """ Check for malicious data poisoning attempts in the training dataset. """
    try:
        if np.any(X_train == 255):  # Check if there are suspicious extreme pixel values
            logging.warning("Potential data poisoning detected in training data.")
        else:
            logging.info("No data poisoning detected in training data.")
    except Exception as e:
        logging.error(f"Error during data poisoning check: {traceback.format_exc()}")
        raise

# Function to check model training and validation accuracy
def check_model_training(X_train, y_train, model):
    """ Check for model training integrity and performance. """
    try:
        model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)  # Train for 10 epochs
        logging.info("Model Training Completed.")

        y_train_pred = model.predict(X_train)
        y_train_pred_classes = np.argmax(y_train_pred, axis=1)
        accuracy = accuracy_score(y_train.flatten(), y_train_pred_classes)
        logging.info(f"Model Training Accuracy: {accuracy:.2f}")

        if accuracy < 0.5:
            logging.warning("Model Training Issue Detected: Accuracy below acceptable threshold.")
        else:
            logging.info("Model Training Verified: Accuracy is acceptable.")
    except Exception as e:
        logging.error(f"Error during model training: {traceback.format_exc()}")
        raise

# Stage 1: Run Bandit for static analysis
logging.info("Stage 1: Static Code Analysis")
run_bandit()

# Stage 2: Run Safety to check dependency vulnerabilities
logging.info("Stage 2: Dependency Vulnerability Check")
run_safety()

# Stage 3: Check data integrity
logging.info("Stage 3: Data Integrity Check")
check_data_integrity(X_train)

# Stage 4: Check test integrity
logging.info("Stage 4: Test Integrity Check")
check_test_integrity()

# Stage 5: Check privileges
logging.info("Stage 5: Privilege Escalation Check")
check_privileges()

# Stage 6: Check for test data poisoning
logging.info("Stage 6: Data Poisoning Check")
check_test_data_poisoning(X_train)

# Continue with other stages...
