import pandas as pd
import hashlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import foolbox as fb
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

# Initialize logging to a file
logging.basicConfig(filename="ml_pipeline_threats.log", level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define a simple neural network model in PyTorch
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)  # Output layer for 3 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Threat detection functions (same as before)
def check_data_integrity(data):
    """ Check data integrity by comparing the hash value before and after loading. """
    data_hash_before = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    logging.info(f"Initial Data Hash: {data_hash_before}")
    
    # Simulate malicious changes
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

def check_dev_catalog():
    """ Simulate checks for the development catalog. """
    model_version = "1.0"
    dataset_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    
    # Simulate a versioning issue
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

def check_prod_catalog():
    """ Simulate checks for the production catalog. """
    # Simulate deployment issues or inconsistencies
    deployment_status = "Success"  # Example: change this to simulate issues
    
    if deployment_status != "Success":
        logging.warning("Production Deployment Failed: Issue detected during model deployment.")
    else:
        logging.info("Production Catalog Verified: Model deployed successfully.")
    
    # Check for model tampering in production
    prod_model_hash = "123456789abcdef"  # Simulated hash for the production model
    current_model_hash = hashlib.sha256(b"current_model").hexdigest()  # Simulate hash check
    
    if prod_model_hash != current_model_hash:
        logging.warning("Model Integrity Compromised: Production model hash mismatch detected.")
    else:
        logging.info("Model Integrity Verified: No tampering detected in production.")

def check_adversarial_attack(X_train, y_train, model):
    """ Check for adversarial vulnerabilities using Cleverhans and Foolbox. """
    
    # Convert dataset to Torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    
    # Foolbox model wrapping
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    
    # FGSM attack using Cleverhans
    adv_examples = fast_gradient_method(fmodel, X_train_tensor, eps=0.1, norm=np.inf)
    
    # Test if adversarial examples change model predictions
    predictions_before = fmodel(X_train_tensor).argmax(axis=-1)
    predictions_after = fmodel(adv_examples).argmax(axis=-1)
    
    if not np.array_equal(predictions_before, predictions_after):
        logging.warning("Adversarial Vulnerability Detected: Model is susceptible to FGSM attack.")
    else:
        logging.info("Adversarial Check Passed: Model is robust to FGSM attack.")

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

logging.info("Stage 2: Data Transformation")
check_data_transformation(X_train_scaled, X_test_scaled)

# Stage 3: Featurization
logging.info("Stage 3: Featurization")
check_featurization(X_train_scaled)

# Stage 4: Model Training with PyTorch
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

# Train the model (simple training loop)
model.train()
for epoch in range(100):  # Train for 100 epochs
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Stage 5: Adversarial Attack Simulation
logging.info("Stage 5: Adversarial Attack Simulation")
check_adversarial_attack(X_train_scaled, y_train, model)

# Stage 6: Development Catalog
logging.info("Stage 6: Development Catalog")
check_dev_catalog()

# Stage 7: Production Catalog
logging.info("Stage 7: Production Catalog")
check_prod_catalog()

# End of pipeline logging
logging.info("ML Pipeline Threat Analysis Completed.")
