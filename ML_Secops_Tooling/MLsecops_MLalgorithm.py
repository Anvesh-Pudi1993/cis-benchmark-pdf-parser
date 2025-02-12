import pandas as pd
import hashlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import os
from cleverhans.attacks.FastGradientMethod import FastGradientMethod
from cleverhans.utils import set_log_level

# Initialize logging to a file
logging.basicConfig(filename="ml_pipeline_threats_ML.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set logging level for CleverHans
set_log_level(logging.WARNING)

# Threat detection functions

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
    model.fit(X_train, y_train)
    logging.info("Model Training Completed.")
    
    # Check model performance
    y_train_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_train_pred)
    logging.info(f"Model Training Accuracy: {accuracy:.2f}")

    if accuracy < 0.5:  # Example threshold for minimal accuracy
        logging.warning("Model Training Issue Detected: Accuracy below acceptable threshold.")
    else:
        logging.info("Model Training Verified: Accuracy is acceptable.")

def check_model_validation(X_test, y_test, model):
    """ Validate the model using the test set. """
    y_test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    logging.info(f"Model Validation Accuracy: {accuracy:.2f}")

    if accuracy < 0.5:  # Example threshold for minimal accuracy
        logging.warning("Model Validation Issue Detected: Accuracy below acceptable threshold.")
    else:
        logging.info("Model Validation Verified: Accuracy is acceptable.")

def check_adversarial_attacks(model, X_train, y_train):
    """ Check for adversarial attacks using CleverHans. """
    # Create adversarial examples using Fast Gradient Method
    fgsm = FastGradientMethod(model)

    # Generate adversarial examples
    X_train_adv = fgsm.generate(X_train, eps=0.1)  # Epsilon defines the attack strength
    y_train_pred_adv = model.predict(X_train_adv)
    
    # Check if the model is fooled by adversarial examples
    if not np.array_equal(y_train, y_train_pred_adv):
        logging.warning("Adversarial Attack Detected: Model predictions differ on adversarial examples.")
    else:
        logging.info("Model is robust to adversarial attacks: Predictions remain consistent.")

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Stage 1: Data Injection (Ingestion)
logging.info("Stage 1: Data Injection")
check_data_integrity(df)
check_data_format(df)

# Stage 2: Data Transformation
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Stage 3: Development Catalog
logging.info("Stage 3: Development Catalog")
model_version = "1.0"
dataset_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
check_dev_catalog(model_version, dataset_hash)

# Stage 4: Model Training
logging.info("Stage 4: Model Training")
model = LogisticRegression()
check_model_training(X_train_scaled, y_train, model)

# Stage 5: Adversarial Attack Check
logging.info("Stage 5: Check for Adversarial Attacks")
check_adversarial_attacks(model, X_train_scaled, y_train)

# Stage 6: Model Validation
logging.info("Stage 6: Model Validation")
check_model_validation(X_test_scaled, y_test, model)
