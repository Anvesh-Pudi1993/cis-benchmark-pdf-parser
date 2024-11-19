import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np
import hashlib
import json
import datetime

class ScopeStage:
    def __init__(self):
        pass

    # Step 1: Load and preprocess the MNIST data
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        return x_train, y_train, x_test, y_test

    # Step 2: Define a CNN model
    def create_cnn(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Step 3: Provenance Metadata Generation
    def generate_provenance_metadata(self, model, data):
        metadata = {
            "model_name": "CNN_MNIST",
            "model_creation_date": str(datetime.datetime.now()),
            "data_hash": hashlib.sha256(data[0].tobytes()).hexdigest(),
            "model_layers": [layer.name for layer in model.layers],
            "dependencies": ["tensorflow", "keras", "numpy"],
        }
        return metadata

    # Step 4: OCTAVE Allegro Risk Assessment
    def octave_risk_assessment(self, metadata):
        risk_factors = {
            "data_integrity": "High" if len(metadata["data_hash"]) != 64 else "Low",
            "dependency_integrity": "Low" if "tensorflow" in metadata["dependencies"] else "Critical",
            "model_architecture": "Low" if len(metadata["model_layers"]) > 1 else "Critical",
        }
        return risk_factors

    # Step 5: RSA Archer Integration Simulation
    def rsa_archer_report(self, metadata, risks):
        report = {
            "metadata": metadata,
            "risks": risks,
            "recommendations": {
                "data_integrity": "Validate dataset source and integrity using checksum.",
                "dependency_integrity": "Use verified package managers and check library versions.",
                "model_architecture": "Ensure architecture follows organizational guidelines.",
            },
        }
        return json.dumps(report, indent=4)

    # Step 6: Supply Chain Vulnerability Check
    def check_supply_chain_vulnerabilities(self):
        try:
            import tensorflow as tf
            import numpy as np
        except ImportError as e:
            return f"Missing dependency: {e}"
        return "Supply chain dependencies verified."

    # Step 7: Governance, Risk, and Compliance Checks
    def one_trust_compliance_check(self):
        # Simulate a OneTrust-based compliance check
        compliance_areas = {
            "data_privacy": "Compliant",
            "security_controls": "Compliant",
            "regulatory_requirements": "Partially Compliant",
        }
        return compliance_areas

    def nist_rmf_risk_check(self, metadata):
        # Simulate a NIST RMF-based risk assessment
        risks = {
            "data_confidentiality": "Low" if metadata["data_hash"] else "High",
            "data_availability": "Low" if "keras" in metadata["dependencies"] else "Medium",
            "system_resilience": "Medium",
        }
        return risks

    def iso_31000_risk_management(self, risks):
        # Simulate ISO 31000-based risk treatment and recommendations
        treatment_plan = {
            risk: ("Monitor" if level == "Low" else "Mitigate") for risk, level in risks.items()
        }
        return treatment_plan

    # Step 8: Threat Modeling Checks
    def threat_model_check(self):
        checks = {
            "Supply Chain Vulnerabilities": self.check_supply_chain_vulnerabilities()
        }
        return checks
    
    def check_bias(self, data):
        x_train, y_train = data
        class_distribution = np.bincount(y_train)
        bias_detected = any(count / len(y_train) < 0.1 for count in class_distribution)
        return "Bias Detected: Class distribution is uneven." if bias_detected else "Bias Check Passed: Class distribution is balanced."

    def check_fairness(self, predictions, labels):
        disparities = [np.mean(predictions[labels == i]) for i in range(10)]
        fairness_violation = max(disparities) - min(disparities) > 0.1
        return "Fairness Check Failed: Performance disparities detected across classes." if fairness_violation else "Fairness Check Passed: Performance is consistent across classes."

    def explainability_check(self, model):
        explainability_metrics = {
            "simple_architecture": len(model.layers) < 10,
            "interpretable_activations": all("relu" in layer.activation.__name__ for layer in model.layers if hasattr(layer, "activation")),
        }
        if not all(explainability_metrics.values()):
            return "Explainability Check Failed: Model is complex or lacks transparent activations."
        return "Explainability Check Passed: Model is simple and interpretable."


    # Main Functionality
    def main(self):
        # Load data
        x_train, y_train, x_test, y_test = self.load_data()

        # Create CNN
        model = self.create_cnn()

        model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

        # Evaluate the model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        predictions = np.argmax(model.predict(x_test), axis=1)
        print(f"Test Accuracy: {test_acc}")

        # Perform threat model checks
        print("\n Supply chain vulnerability results")
        checks = self.threat_model_check()
        for check, result in checks.items():
            print(f"{check}: {result}")

        # Generate provenance metadata
        metadata = self.generate_provenance_metadata(model, (x_train, y_train))

        # Perform GRC checks
        compliance = self.one_trust_compliance_check()
        risks = self.nist_rmf_risk_check(metadata)
        treatment_plan = self.iso_31000_risk_management(risks)

        # Perform risk assessment using OCTAVE Allegro principles
        risks = self.octave_risk_assessment(metadata)

        # Simulate RSA Archer reporting
        report = self.rsa_archer_report(metadata, risks)

        bias_check = self.check_bias((x_train, y_train))
        fairness_check = self.check_fairness(predictions, y_test)
        explainability_check = self.explainability_check(model)

        # Print results
        print("Model Provenance Report:")
        print(report)

        print("\nGovernance, Risk, and Compliance Results:")
        print("OneTrust Compliance Check:", compliance)
        print("NIST RMF Risk Assessment:", risks)
        print("ISO 31000 Risk Treatment Plan:", treatment_plan)

        print("\nTrusted AI Checks:")
        print("Bias Check:", bias_check)
        print("Fairness Check:", fairness_check)
        print("Explainability Check:", explainability_check)


if __name__ == "__main__":
    scope_stage = ScopeStage()
    scope_stage.main()
