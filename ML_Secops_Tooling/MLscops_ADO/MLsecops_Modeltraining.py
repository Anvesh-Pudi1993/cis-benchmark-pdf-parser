import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
import matplotlib.pyplot as plt
import json
import pandas as pd

class TrustedAI:
    def __init__(self):
        self.model = None

    def load_data(self):
        """
        Load and preprocess the MNIST dataset.
        """
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

        # Create a binary classification task for fairness (digits 0 and 1)
        train_filter = (y_train == 0) | (y_train == 1)
        test_filter = (y_test == 0) | (y_test == 1)
        x_train, y_train = x_train[train_filter], y_train[train_filter]
        x_test, y_test = x_test[test_filter], y_test[test_filter]

        return x_train, y_train, x_test, y_test

    def create_model(self):
        """
        Create and compile a CNN model for binary classification.
        """
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid")  # Binary classification
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def train_model(self, x_train, y_train):
        """
        Train the CNN model.
        """
        self.model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

    def fairness_check(self, x_test, y_test):
        """
        Perform bias and fairness checks using AI Fairness 360 and Fairlearn.
        """
        # Predict using the trained model
        y_pred = (self.model.predict(x_test) > 0.5).astype(int).flatten()

        # Convert data to Pandas DataFrame for AIF360
        data = {
            "label": y_test.flatten(),
            "predicted": y_pred.flatten(),
            "protected_attribute": y_test.flatten(),  # Example: Using the label as a protected attribute
        }
        df = pd.DataFrame(data)

        # Create a BinaryLabelDataset
        test_dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df,
            label_names=["predicted"],
            protected_attribute_names=["protected_attribute"]
        )

        # AI Fairness 360 Metrics
        metric = BinaryLabelDatasetMetric(
            test_dataset,
            privileged_groups=[{"protected_attribute": 1}],  # Example: 1 as privileged
            unprivileged_groups=[{"protected_attribute": 0}]  # Example: 0 as unprivileged
        )

        # Fairlearn Metrics
        metric_frame = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "accuracy": lambda y_true, y_pred: np.mean(y_true == y_pred),
            },
            y_true=y_test.flatten(),
            y_pred=y_pred.flatten(),
            sensitive_features=y_test.flatten()  # Example: Using the label as a proxy for sensitive feature
        )
        demographic_parity = demographic_parity_difference(
            y_test.flatten(), y_pred.flatten(), sensitive_features=y_test.flatten()
        )

        print("Fairness Check Completed.")
        return {
            "AI Fairness 360 Metrics": {
                "Disparate Impact": float(metric.disparate_impact())  # Ensure the value is JSON-serializable
            },
            "Fairlearn Metrics": {
                "Metric Frame Summary": metric_frame.overall.to_dict(),  # Convert to a JSON-serializable format
                "Demographic Parity Difference": float(demographic_parity)  # Ensure the value is JSON-serializable
            }
        }


    def explainability_check(self, x_sample):
    # Ensure x_sample is a TensorFlow tensor and enable gradient tracking
        x_sample = tf.convert_to_tensor(x_sample, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_sample)  # Now it can be tracked
            predictions = self.model(x_sample)

        grads = tape.gradient(predictions, x_sample)
        saliency_map = np.abs(grads).mean(axis=-1)

        # Visualize the saliency map
        plt.imshow(saliency_map[0], cmap='hot')
        plt.title("Saliency Map")
        plt.colorbar()
        plt.show()

        return "Saliency map generated for explainability."

    def main(self):
        """
        Main pipeline for Trusted AI checks.
        """
        # Load data
        x_train, y_train, x_test, y_test = self.load_data()

        # Create and train the model
        self.create_model()
        self.train_model(x_train, y_train)

        # Perform fairness checks
        fairness_report = self.fairness_check(x_test, y_test)
        print("Fairness Report:\n", json.dumps(fairness_report, indent=4))

        # Perform explainability checks
        self.explainability_check(x_test[:1])  # Use the first test sample


if __name__ == "__main__":
    trusted_ai_pipeline = TrustedAI()
    trusted_ai_pipeline.main()
