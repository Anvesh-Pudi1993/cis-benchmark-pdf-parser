import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import hashlib
import json

class DataPreparationStage:
    def __init__(self):
        pass

    # Step 1: Load and preprocess the MNIST data
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        return x_train, y_train, x_test, y_test

    # Step 2: Apache Ranger-based Data Access Policy Check
    def apache_ranger_check(self, user_role, resource):
        # Simulated access control rules
        access_policies = {
            "admin": ["read", "write", "delete"],
            "data_scientist": ["read", "write"],
            "viewer": ["read"]
        }

        if user_role not in access_policies:
            return "Access Denied: Role not recognized."

        allowed_actions = access_policies[user_role]
        resource_access = {
            "MNIST_data": "read",
            "Model_parameters": "write"
        }

        resource_action = resource_access.get(resource, "none")
        if resource_action not in allowed_actions:
            return f"Access Denied: {user_role} cannot perform '{resource_action}' on {resource}."
        return f"Access Granted: {user_role} can perform '{resource_action}' on {resource}."

    # Step 3: Okta-based User Authentication Check
    def okta_authentication_check(self, user_credentials):
        # Simulated authentication mechanism
        valid_users = {
            "admin_user": "admin_password",
            "data_scientist_user": "ds_password",
            "viewer_user": "viewer_password"
        }

        username = user_credentials.get("username")
        password = user_credentials.get("password")

        if valid_users.get(username) == password:
            return f"Authentication Successful: Welcome {username}."
        return "Authentication Failed: Invalid username or password."

    # Step 4: Data Integrity Check
    def data_integrity_check(self, data):
        # Generate SHA-256 hash of the dataset
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()
        # Simulated hash value for validation
        expected_hash = "b1946ac92492d2347c6235b4d2611184e4f78dbdfc8c3fbb5f53b52f3f7cdd54"

        if data_hash == expected_hash:
            return "Data Integrity Verified: Hash matches the expected value."
        return "Data Integrity Compromised: Hash does not match."

    # Step 5: Supply Chain Vulnerability Report
    def generate_supply_chain_report(self, apache_ranger_result, okta_result, integrity_result):
        report = {
            "apache_ranger_check": apache_ranger_result,
            "okta_authentication_check": okta_result,
            "data_integrity_check": integrity_result
        }
        return json.dumps(report, indent=4)

    # Main Functionality
    def main(self):
        # Simulate user credentials and roles
        user_credentials = {"username": "data_scientist_user", "password": "ds_password"}
        user_role = "data_scientist"

        # Load data
        x_train, y_train, x_test, y_test = self.load_data()

        # Perform Apache Ranger-based data access policy check
        apache_ranger_result = self.apache_ranger_check(user_role, "MNIST_data")

        # Perform Okta-based user authentication check
        okta_result = self.okta_authentication_check(user_credentials)

        # Perform data integrity check
        integrity_result = self.data_integrity_check(x_train)

        # Generate supply chain vulnerability report
        supply_chain_report = self.generate_supply_chain_report(apache_ranger_result, okta_result, integrity_result)

        # Print the report
        print("Supply Chain Vulnerability Report at data preparation stage:")
        print(supply_chain_report)


if __name__ == "__main__":
    preparation_stage = DataPreparationStage()
    preparation_stage.main()
