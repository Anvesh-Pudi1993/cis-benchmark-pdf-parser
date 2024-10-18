import os
import pickle
import hashlib
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import logging
import boto3

# Set up logging
logging.basicConfig(filename='dev_catalog_threat_check.log', level=logging.INFO)

# 1. Load Iris dataset
def load_iris_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=["target"])
    return X, y

# 2. Train a Logistic Regression model (supervised ML)
def train_model(X, y):
    model = LogisticRegression(max_iter=200)
    model.fit(X, y.values.ravel())
    return model

# 3. Serialize and encrypt the model
def serialize_and_encrypt_model(model, encryption_key):
    fernet = Fernet(encryption_key)
    serialized_model = pickle.dumps(model)
    encrypted_model = fernet.encrypt(serialized_model)
    return encrypted_model

# 4. Save encrypted model to local storage (simulating storage in dev catalog)
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

# 6. Check for data poisoning (outlier detection using IsolationForest)
def check_data_poisoning(X):
    clf = IsolationForest(contamination=0.01)
    clf.fit(X)
    outliers = clf.predict(X)
    num_outliers = (outliers == -1).sum()
    if num_outliers > 0:
        logging.warning(f"Data Poisoning Alert: {num_outliers} outliers detected in feature set")
    else:
        logging.info("No significant outliers detected (No data poisoning)")

# 7. Check S3 bucket access control (Unauthorized Access)
def check_s3_access(bucket_name, s3_client):
    try:
        acl = s3_client.get_bucket_acl(Bucket=bucket_name)
        for grant in acl['Grants']:
            if 'AllUsers' in grant['Grantee'].get('URI', ''):
                logging.warning(f"Public access detected for bucket {bucket_name}")
    except Exception as e:
        logging.error(f"Failed to check S3 ACL for bucket {bucket_name}: {e}")

# Main function to run all checks
def run_threat_vector_checks():
    # Load and train model
    X, y = load_iris_data()
    model = train_model(X, y)

    # Encrypt model
    encryption_key = Fernet.generate_key()
    encrypted_model = serialize_and_encrypt_model(model, encryption_key)
    save_to_local("encrypted_model.pkl", encrypted_model)

    # Hash model to verify integrity
    model_hash = hashlib.sha256(pickle.dumps(model)).hexdigest()
    verify_model_integrity("encrypted_model.pkl", model_hash)

    # Check for data poisoning
    check_data_poisoning(X)

    # S3 access check (if using S3 for storage)
    s3_client = boto3.client('s3')
    check_s3_access('my-dev-bucket', s3_client)

# Run all threat vector checks
if __name__ == "__main__":
    run_threat_vector_checks()
