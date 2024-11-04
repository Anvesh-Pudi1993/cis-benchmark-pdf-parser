# Import required libraries
import logging
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from deepchecks.tabular import Dataset
# from deepchecks.tabular.checks import TrainTestFeatureDrift, TrainTestLabelDrift
# Import the correct check for feature drift
from deepchecks.tabular.checks import FeatureDrift, LabelDrift
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
from tensorflow import keras
import tensorflow as tf
# Initialize logging
logging.basicConfig(filename="mlops_threat_vectors.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Define the categories we want to keep
categories_to_keep = [0, 1, 2]  # airplane, automobile, bird

# Function to filter data by categories
def filter_data(images, labels, categories):
    mask = np.isin(labels, categories)  # Create a mask for the desired categories
    filtered_images = images[mask.flatten()]
    filtered_labels = labels[mask.flatten()]
    # Re-labeling the filtered labels to be 0, 1, 2 for the selected categories
    filtered_labels = np.array([categories.index(label[0]) for label in filtered_labels])
    return filtered_images, filtered_labels

# Step 1: Filter the data for the selected categories
logging.info("Step 1: Filtering data for selected categories")
train_images_filtered, train_labels_filtered = filter_data(train_images, train_labels, categories_to_keep)
test_images_filtered, test_labels_filtered = filter_data(test_images, test_labels, categories_to_keep)

# Function to log data poisoning check
def check_data_poisoning(data):
    # Example poisoning detection logic (Placeholder for more complex checks)
    if np.mean(data) < 50:  # Example condition for detection
        logging.warning("Data poisoning suspected: Mean pixel value lower than expected.")
    else:
        logging.info("No data poisoning detected.")

# EDA: Check for anomalies, data poisoning, and data distribution
def run_eda_checks(images, labels):
    logging.info("Running EDA Checks")

    # Check for missing data
    logging.info(f"Missing values in data: {np.isnan(images).sum()}")

    # Check for class imbalance
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
    logging.info(f"Class distribution: {dict(zip(unique, counts))}")

    # Check for outliers using Z-scores
    flattened_images = images.reshape(images.shape[0], -1)  # Flatten the images for easier processing
    z_scores = np.abs(stats.zscore(flattened_images, axis=1))  # Z-score along the features axis
    outliers = np.where(z_scores > 3)
    logging.info(f"Number of outliers detected: {len(outliers[0])}")

    # Data poisoning check
    check_data_poisoning(images)

# Step 2: Run EDA checks for train and test sets
logging.info("Step 2: Running EDA checks for training data")
run_eda_checks(train_images_filtered, train_labels_filtered)


# Step 3: Feature Drift & Label Drift Check using Deepchecks
logging.info("Step 3: Running Feature and Label Drift checks")

# Reshape images to 2D
train_images_flat = train_images_filtered.reshape((train_images_filtered.shape[0], -1))  # Shape (num_samples, 3072)
test_images_flat = test_images_filtered.reshape((test_images_filtered.shape[0], -1))  # Shape (num_samples, 3072)

# Create Dataset objects (No categorical features in this case)
train_dataset = Dataset(train_images_flat, train_labels_filtered)
test_dataset = Dataset(test_images_flat, test_labels_filtered)

# Run the new Feature Drift check
feature_drift_check = FeatureDrift().run(train_dataset, test_dataset)
# logging.info(f"Feature Drift Check Result: {feature_drift_check.value}")

# Run the new Label Drift check
label_drift_check = LabelDrift().run(train_dataset, test_dataset)
# logging.info(f"Label Drift Check Result: {label_drift_check.value}")

# Log full details of the checks
# logging.info(f"Feature Drift Check Details: {str(feature_drift_check)}")
# logging.info(f"Label Drift Check Details: {str(label_drift_check)}")

# Step 4: Model Training - Check for adversarial vulnerabilities
logging.info("Step 4: Checking model training for adversarial vulnerabilities")
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation='softmax')  # Output layer should have 3 units for the 3 classes
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Wrap the model using ART classifier
classifier = TensorFlowV2Classifier(model=model, nb_classes=3, input_shape=(32, 32, 3), 
                                     loss_object=tf.keras.losses.SparseCategoricalCrossentropy())

# Create an adversarial attack
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=test_images_filtered)

# Predict on adversarial examples
predictions = classifier.predict(x_test_adv)

# Calculate accuracy
predicted_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_labels == test_labels_filtered)  # Calculate accuracy manually
logging.info(f"Model accuracy on adversarial samples: {accuracy}")

# Step 5: Final log after threat checks
logging.info("Threat vector analysis completed. Logs saved.")
