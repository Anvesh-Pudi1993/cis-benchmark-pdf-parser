import pandas as pd
import hashlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import os
import tensorflow as tf  # TensorFlow for Keras model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from cleverhans.tf2.attacks import fast_gradient_method
from cleverhans.utils import set_log_level

# Initialize logging to a file
logging.basicConfig(filename="ml_pipeline_threats_V2.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set logging level for CleverHans
set_log_level(logging.WARNING)

# Threat detection functions...

def check_data_integrity(data):
    """ Check data integrity by comparing the hash value before and after loading. """
    data_hash_before = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    logging.info(f"Initial Data Hash: {data_hash_before}")
    
    # Simulate malicious changes (e.g., data poisoning)
    data_copy = data.copy()
    data_copy.iloc[0, 0] = 999  # Simulate Data Poisoning
    data_hash_after = hashlib.sha256(data_copy.to_csv(index=False).encode()).hexdigest()
    
    if data_hash_before != data_hash_after:
        logging.warning("Data Integrity Compromised: Hash mismatch detected after ingestion.")
    else:
        logging.info("Data Integrity Verified: No issues with the data.")

def check_data_format(data):
    """ Check for missing values, duplicates, invalid formats, and schema issues. """
    if data.isnull().values.any():
        logging.warning("Missing values detected in the dataset.")
    if data.duplicated().any():
        logging.warning("Duplicate rows detected in the dataset.")
    else:
        logging.info("Data Format Check: No missing or duplicate values detected.")

    # Check for schema issues
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    if list(data.columns) != expected_columns:
        logging.warning("Schema Mismatch Detected: Column names do not match the expected schema.")
    else:
        logging.info("Schema Check Passed: No schema issues.")

def check_data_transformation(X_train, X_test):
    """ Check for data leakage or suspicious transformations. """
    if np.array_equal(X_train, X_test):
        logging.warning("Data Leakage Detected: Training and test data are identical.")
    else:
        logging.info("Data Transformation Verified: No data leakage detected.")

    # Check for invalid transformations
    if np.any(X_train == np.inf) or np.any(X_train == -np.inf):
        logging.warning("Invalid Transformation Detected: Infinite values in training data.")
    if np.any(np.isnan(X_train)):
        logging.warning("Invalid Transformation Detected: NaN values in training data.")
    else:
        logging.info("Transformation Check Passed: No invalid transformations.")

def check_featurization(X):
    """ Check for feature manipulation, leakage, or suspicious feature engineering. """
    feature_variance = np.var(X, axis=0)
    if np.any(feature_variance == 0):
        logging.warning("Feature Manipulation Detected: Some features have zero variance.")
    else:
        logging.info("Featurization Verified: No suspicious feature manipulation detected.")
    
    # Bias detection (naive method)
    if np.max(X) > 10:  # Example threshold for feature scaling issues
        logging.warning("Feature Scaling Issue Detected: Feature values are abnormally high.")
    else:
        logging.info("Feature Scaling Check Passed.")

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
    user = os.getenv('USER')  # Example check for unauthorized access
    if user != "authorized_user":
        logging.warning(f"Unauthorized Access Detected: {user} attempted to access the development catalog.")
    else:
        logging.info("Development Catalog Access Verified: Authorized access detected.")

def check_model_training(X_train, y_train, model):
    """ Check for model training integrity and performance. """
    model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=0)  # Ensure y_train is integers
    logging.info("Model Training Completed.")
    
    # Check model performance
    y_train_pred = model.predict(X_train)
    y_train_pred_classes = np.argmax(y_train_pred, axis=1)  # Convert predictions to class labels
    accuracy = accuracy_score(y_train, y_train_pred_classes)
    logging.info(f"Model Training Accuracy: {accuracy:.2f}")

    if accuracy < 0.5:  # Example threshold for minimal accuracy
        logging.warning("Model Training Issue Detected: Accuracy below acceptable threshold.")
    else:
        logging.info("Model Training Verified: Accuracy is acceptable.")


def check_model_validation(X_test, y_test, model):
    """ Validate the model using the test set. """
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)  # Convert predictions to class labels
    accuracy = accuracy_score(y_test, y_test_pred_classes)  # Directly compare y_test with integer labels
    logging.info(f"Model Validation Accuracy: {accuracy:.2f}")

    if accuracy < 0.5:  # Example threshold for minimal accuracy
        logging.warning("Model Validation Issue Detected: Accuracy below acceptable threshold.")
    else:
        logging.info("Model Validation Verified: Accuracy is acceptable.")

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method  # Adjust based on your version

def check_adversarial_attacks(model, X_train, y_train):
    """ Check for adversarial attacks using CleverHans. """
    # Convert to TensorFlow tensor
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)  
    
    # Create a function for model predictions
    def get_logits(x):
        return model(x, training=False)  # Ensure model is in inference mode

    # Generate adversarial examples using Fast Gradient Method
    X_train_adv = fast_gradient_method(get_logits, X_train_tensor, eps=0.1, norm=np.inf)  # Epsilon defines the attack strength
    
    # Get predictions for adversarial examples
    y_train_pred_adv = model(X_train_adv)
    y_train_pred_adv_classes = np.argmax(y_train_pred_adv, axis=1)  # Convert predictions to class labels

    # Check if the model is fooled by adversarial examples
    if not np.array_equal(y_train, y_train_pred_adv_classes):  # Compare with original y_train
        logging.warning("Adversarial Attack Detected: Model predictions differ on adversarial examples.")
    else:
        logging.info("Model is robust to adversarial attacks: Predictions remain consistent.")

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split the dataset
X = df.drop('target', axis=1)
y = df['target'].values  # Ensure y is an integer array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for the Iris dataset
])

# Compile the model using sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Stage 1: Data Injection (Ingestion)
logging.info("Stage 1: Data Injection")
check_data_integrity(df)
check_data_format(df)

# Stage 2: Data Transformation
logging.info("Stage 2: Data Transformation")
check_data_transformation(X_train_scaled, X_test_scaled)

# Stage 3: Feature Engineering
logging.info("Stage 3: Feature Engineering")
check_featurization(X_train_scaled)

# Stage 4: Model Training with TensorFlow Keras
logging.info("Stage 4: Model Training")
check_model_training(X_train_scaled, y_train, model)  # Pass y_train directly without conversion

# Stage 5: Adversarial Attack Check
logging.info("Stage 5: Check for Adversarial Attacks")
check_adversarial_attacks(model, X_train_scaled, y_train)  # Pass y_train directly without conversion

# Stage 6: Model Validation
logging.info("Stage 6: Model Validation")
check_model_validation(X_test_scaled, y_test, model)  # Pass y_test directly without conversion

# Stage 7: Development Catalog Check
dataset_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
check_dev_catalog("1.0", dataset_hash)  # Pass the expected version
