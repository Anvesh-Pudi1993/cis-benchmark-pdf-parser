import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import hashlib
import os
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.utils import set_log_level

# Initialize logging to a file
logging.basicConfig(filename="ml_pipeline_threats_V3.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set logging level for CleverHans
set_log_level(logging.WARNING)

# Load and preprocess the CIFAR-10 dataset
logging.info("Loading CIFAR-10 dataset")
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

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
    data_hash_before = hashlib.sha256(data.tobytes()).hexdigest()
    logging.info(f"Initial Data Hash: {data_hash_before}")

    # Simulate malicious changes (data poisoning)
    data_copy = np.copy(data)
    data_copy[0, 0, 0, 0] = 255  # Simulate Data Poisoning
    data_hash_after = hashlib.sha256(data_copy.tobytes()).hexdigest()

    if data_hash_before != data_hash_after:
        logging.warning("Data Integrity Compromised: Hash mismatch detected after ingestion.")
    else:
        logging.info("Data Integrity Verified: No issues with the data.")

# Function to check data transformation
def check_data_transformation(X_train, X_test):
    """ Check for data leakage, invalid transformations, NaN, and infinite values. """
    if np.array_equal(X_train, X_test):
        logging.warning("Data Leakage Detected: Training and test data are identical.")
    else:
        logging.info("Data Transformation Verified: No data leakage detected.")

    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        logging.warning("Invalid Transformation Detected: NaN values present in the data.")
    if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
        logging.warning("Invalid Transformation Detected: Infinite values present in the data.")
    else:
        logging.info("Transformation Check Passed: No invalid values detected.")

# Function to check featurization
def check_featurization(X):
    """ Check for feature manipulation, zero variance, or suspicious feature engineering. """
    feature_variance = np.var(X, axis=0)
    if np.any(feature_variance == 0):
        logging.warning("Feature Manipulation Detected: Some features have zero variance.")
    else:
        logging.info("Featurization Verified: No suspicious feature manipulation detected.")
    
    # Check for any unusual pixel values after normalization
    if np.max(X) > 1.0 or np.min(X) < 0.0:
        logging.warning("Feature Scaling Issue Detected: Feature values out of expected range [0,1].")
    else:
        logging.info("Feature Scaling Check Passed: Feature values within expected range.")

# Function to check model training and validation accuracy
def check_model_training(X_train, y_train, model):
    """ Check for model training integrity and performance. """
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

# Function to check model validation
def check_model_validation(X_test, y_test, model):
    """ Validate the model using the test set. """
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    accuracy = accuracy_score(y_test.flatten(), y_test_pred_classes)
    logging.info(f"Model Validation Accuracy: {accuracy:.2f}")

    if accuracy < 0.5:
        logging.warning("Model Validation Issue Detected: Accuracy below acceptable threshold.")
    else:
        logging.info("Model Validation Verified: Accuracy is acceptable.")

# Function to check adversarial attacks
def check_adversarial_attacks(model, X_train, y_train):
    """ Check for adversarial attacks using CleverHans. """
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)

    def get_logits(x):
        return model(x, training=False)

    X_train_adv = fast_gradient_method(get_logits, X_train_tensor, eps=0.1, norm=np.inf)
    
    y_train_pred_adv = model.predict(X_train_adv)
    y_train_pred_adv_classes = np.argmax(y_train_pred_adv, axis=1)

    if not np.array_equal(y_train.flatten(), y_train_pred_adv_classes):
        logging.warning("Adversarial Attack Detected: Model predictions differ on adversarial examples.")
    else:
        logging.info("Model is robust to adversarial attacks: Predictions remain consistent.")

# Function to check development catalog
def check_dev_catalog(model_version, dataset_hash):
    """ Simulate checks for the development catalog. """
    expected_version = "1.0"

    logging.info(f"Model Version: {model_version}")
    logging.info(f"Dataset Hash: {dataset_hash}")

    if model_version != expected_version:
        logging.warning("Version Inconsistency Detected: Model version does not match expected version.")
    else:
        logging.info("Development Catalog Verified: Model and data versions are consistent.")

    # Simulate unauthorized access
    user = os.getenv('USER')
    if user != "authorized_user":
        logging.warning(f"Unauthorized Access Detected: {user} attempted to access the development catalog.")
    else:
        logging.info("Development Catalog Access Verified: Authorized access detected.")

# Define a Convolutional Neural Network (CNN) model for 3 classes
logging.info("Building CNN model")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes output
])

# Compile the model
logging.info("Compiling model")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Stage 1: Data Integrity Check
logging.info("Stage 1: Data Injection")
check_data_integrity(X_train)

# Stage 2: Data Transformation Check
logging.info("Stage 2: Data Transformation")
check_data_transformation(X_train, X_test)

# Stage 3: Feature Engineering Check (Featurization)
logging.info("Stage 3: Feature Engineering")
check_featurization(X_train)

# Stage 4: Model Training
logging.info("Stage 4: Model Training")
check_model_training(X_train, y_train, model)

# Stage 5: Check for Adversarial Attacks
logging.info("Stage 5: Check for Adversarial Attacks")
check_adversarial_attacks(model, X_train, y_train)

# Stage 6: Model Validation
logging.info("Stage 6: Model Validation")
check_model_validation(X_test, y_test, model)

# Stage 7: Development Catalog Check
logging.info('stage 7: check for dev catalog')
dataset_hash = hashlib.sha256(X_train.tobytes()).hexdigest()
check_dev_catalog("1.0", dataset_hash)
