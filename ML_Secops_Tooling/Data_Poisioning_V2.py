import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from art.defences.detector.poison import ClusteringDefense
from art.estimators.classification import KerasClassifier
from art.utils import load_cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test), _, _ = load_cifar10()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the CNN model
model = create_cnn_model()
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Wrap the model with ART's KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Use ClusteringDefense to analyze data poisoning
def check_data_poisoning_with_clustering(classifier, x_train, y_train):
    # Initialize ClusteringDefense
    clustering_defense = ClusteringDefense(classifier, x_train, y_train)

    # Perform feature clustering to identify potential poisons
    print("Extracting features and running clustering analysis...")
    clustering_defense.detect_poison(n_clusters=10)  # Number of clusters can be adjusted based on the dataset

    # Get summary of clusters
    report = clustering_defense.get_clusters()

    # Analyze the clusters for potential poisoning
    is_poisoned = clustering_defense.analyze_clusters(report)
    
    if is_poisoned:
        print("Data poisoning detected!")
    else:
        print("No data poisoning detected.")

# Run data poisoning check
check_data_poisoning_with_clustering(classifier, x_train, y_train)
