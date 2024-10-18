import os
import pickle
import hashlib
import pandas as pd
import numpy as np
import logging
from cryptography.fernet import Fernet
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import boto3

# Configure logging
logging.basicConfig(filename='ml_threat_vectors.log', level=logging.INFO)

# 1. Load Iris dataset
def load_iris_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=["target"])
    return X, y

# 2. Train models (Supervised and Unsupervised)
def train_models(X, y):
    # Logistic Regression for supervised learning
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y.values.ravel())

    # KMeans for unsupervised learning
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    return clf, kmeans

# 3. Serialize and encrypt models
def serialize_and_encrypt_model(model, encryption_key):
    fernet = Fernet(encryption_key)
    # Serialize model
    serialized_model = pickle.dumps(model)
    # Encrypt model
    encrypted_model = fernet.encrypt(serialized_model)
    return encrypted_model

# 4. Save encrypted models and features to S3
# def save_to_s3(bucket_name, file_name, data, s3_client):
#     try:
#         s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=data)
#         logging.info(f"Successfully uploaded {file_name} to S3 bucket {bucket_name}.")
#     except Exception as e:
#         logging.error(f"Failed to upload {file_name} to S3: {e}")

# # 5. Check S3 encryption and public access
# def check_s3_encryption_and_access(bucket_name, s3_client):
#     try:
#         encryption = s3_client.get_bucket_encryption(Bucket=bucket_name)
#         logging.info(f"S3 Bucket Encryption: {encryption}")
#     except Exception as e:
#         logging.warning(f"No encryption found for bucket {bucket_name}: {e}")

#     try:
#         acl = s3_client.get_bucket_acl(Bucket=bucket_name)
#         for grant in acl['Grants']:
#             if 'AllUsers' in grant['Grantee'].get('URI', ''):
#                 logging.warning(f"Public access detected for bucket {bucket_name}!")
#     except Exception as e:
#         logging.error(f"Failed to check ACL for bucket {bucket_name}: {e}")

# 6. Check for Data Poisoning (basic outlier detection)
def check_for_data_poisoning(X):
    z_scores = np.abs((X - X.mean()) / X.std())
    outliers = (z_scores > 3).sum().sum()
    if outliers > 0:
        logging.warning(f"Data Poisoning Alert: {outliers} outliers detected in feature set.")
    else:
        logging.info("No significant outliers detected (No data poisoning detected).")

# 7. Verify model integrity (hash comparison)
def verify_model_integrity(saved_model_path, expected_hash):
    sha256_hash = hashlib.sha256()
    try:
        with open(saved_model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        current_hash = sha256_hash.hexdigest()
        if current_hash == expected_hash:
            logging.info(f"Model integrity verified for {saved_model_path}.")
        else:
            logging.warning(f"Model integrity check failed for {saved_model_path}.")
    except Exception as e:
        logging.error(f"Error verifying model integrity: {e}")

# Main function to check for threat vectors
def run_threat_vector_checks():
    # Load and train models
    X, y = load_iris_data()
    clf, kmeans = train_models(X, y)

    # Encryption key generation
    encryption_key = Fernet.generate_key()

    # Serialize and encrypt models
    encrypted_clf = serialize_and_encrypt_model(clf, encryption_key)
    encrypted_kmeans = serialize_and_encrypt_model(kmeans, encryption_key)

    # Save models and features to S3
    s3_client = boto3.client('s3')
    bucket_name = 'ml-model-storage'
    # save_to_s3(bucket_name, 'logistic_regression_model.pkl', encrypted_clf, s3_client)
    # save_to_s3(bucket_name, 'kmeans_model.pkl', encrypted_kmeans, s3_client)

    # Check S3 encryption and access
    # check_s3_encryption_and_access(bucket_name, s3_client)

    # Check for data poisoning
    check_for_data_poisoning(X)

    # Verify model integrity (for demonstration purposes, let's assume a hash)
    expected_hash = hashlib.sha256(pickle.dumps(clf)).hexdigest()
    verify_model_integrity('logistic_regression_model.pkl', expected_hash)

# Run the checks
if __name__ == "__main__":
    run_threat_vector_checks()
