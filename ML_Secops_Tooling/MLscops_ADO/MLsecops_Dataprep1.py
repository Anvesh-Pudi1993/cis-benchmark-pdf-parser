import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import hashlib
import json


class DataPreparationStageWithAdversarialChecks:
    def __init__(self):
        pass

    # Step 1: Load and preprocess the MNIST data
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        return x_train, y_train, x_test, y_test

    # Step 2: ARX Data Anonymization Check
    def arx_data_anonymization(self, data):
        """
        Simulated ARX Data Anonymization: Applies generalization by truncating pixel values to reduce information leakage.
        """
        anonymized_data = np.floor(data * 10) / 10  # Generalize pixel values to one decimal place
        generalization_success = not np.array_equal(data, anonymized_data)
        return anonymized_data, {
            "generalization_applied": generalization_success,
            "information_loss": float(np.mean(np.abs(data - anonymized_data)))  # Convert to float
        }

    # Step 3: HashiCorp Vault Simulation for Secure Key Management
    def hashi_vault_key_management(self, operation, key=None, data=None):
        """
        Simulates HashiCorp Vault operations for secure key management.
        - For 'encrypt': Generates a mock encryption key and encrypts the data.
        - For 'decrypt': Decrypts the data using the mock encryption key.
        """
        mock_key = hashlib.sha256(b"secure_mock_key").hexdigest()

        if operation == "encrypt":
            encrypted_data = hashlib.sha256(data.tobytes() + mock_key.encode()).hexdigest()
            return encrypted_data, mock_key

        if operation == "decrypt":
            if key:
                return "Decryption Successful"  # Simulate decryption success
            else:
                return "Decryption Failed: Key missing."

    # Step 4: Adversarial Defense Check (Noise Injection)
    def adversarial_defense_check(self, x_train):
        """
        Simulates adversarial noise injection and evaluates model robustness.
        Adds small noise to data and checks if the pixel values remain in range [0, 1].
        """
        noise = np.random.normal(0, 0.1, x_train.shape)  # Add Gaussian noise
        noisy_data = np.clip(x_train + noise, 0, 1)  # Ensure values are within valid range

        noise_stats = {
            "mean_noise": float(np.mean(noise)),  # Convert to float
            "max_noise": float(np.max(noise)),    # Convert to float
            "min_noise": float(np.min(noise))     # Convert to float
        }

        return noisy_data, noise_stats

    # Step 5: Adversarial ML Report Generation
    def generate_adversarial_ml_report(self, anonymization_result, vault_result, defense_result):
        # Ensure all fields in the report are JSON serializable
        report = {
            "arx_anonymization": anonymization_result,
            "hashicorp_vault": vault_result,
            "adversarial_defense": {
                "noise_stats": defense_result["noise_stats"],
                "robustness_verified": bool(defense_result["robustness_verified"])
            }
        }
        return json.dumps(report, indent=4)

    # Main Functionality
    def main(self):
        # Load data
        x_train, y_train, x_test, y_test = self.load_data()

        # ARX Data Anonymization Check
        anonymized_data, anonymization_result = self.arx_data_anonymization(x_train)

        # HashiCorp Vault Key Management Simulation
        encrypted_data, encryption_key = self.hashi_vault_key_management("encrypt", data=x_train)
        vault_result = {
            "encryption_key": encryption_key,
            "encrypted_data_sample": encrypted_data[:64]  # Display a sample of the encrypted data
        }

        # Adversarial Defense Check
        noisy_data, noise_stats = self.adversarial_defense_check(x_train)
        defense_result = {
            "noise_stats": noise_stats,
            "robustness_verified": np.all((noisy_data >= 0) & (noisy_data <= 1))
        }

        # Generate Adversarial ML Report
        adversarial_ml_report = self.generate_adversarial_ml_report(anonymization_result, vault_result, defense_result)

        # Print the report
        print("Adversarial ML Check Report:")
        print(adversarial_ml_report)


if __name__ == "__main__":
    preparation_stage = DataPreparationStageWithAdversarialChecks()
    preparation_stage.main()
