import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1 import disable_eager_execution
from art.estimators.classification import KerasClassifier
from art.utils import load_cifar10
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Disable eager execution for ART compatibility
disable_eager_execution()

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test), _, _ = load_cifar10()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple CNN model in a way compatible with TensorFlow's graph mode
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

# Train the CNN model
model = create_cnn_model()
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Wrap the model with ART's KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Feature Extraction for Poison Detection
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extractor.predict(x_train)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)

# Use KMeans for clustering
n_clusters = 10  # CIFAR-10 has 10 classes
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_pca)

# Analyze cluster distribution and check for potential data poisoning
def check_data_poisoning(clusters, y_labels):
    cluster_counts = np.zeros((n_clusters, 10))  # 10 classes in CIFAR-10
    
    # Count labels in each cluster
    for i, cluster_id in enumerate(clusters):
        label = np.argmax(y_labels[i])
        cluster_counts[cluster_id, label] += 1
    
    # Identify clusters with mixed labels
    for i, count in enumerate(cluster_counts):
        if np.max(count) < 0.5 * np.sum(count):  # Adjust threshold based on poisoning severity
            return True  # Data poisoning detected
    
    return False  # No data poisoning detected

# Run analysis to check for data poisoning
is_poisoned = check_data_poisoning(clusters, y_train)

# Print result
if is_poisoned:
    print("Data poisoning detected!")
else:
    print("No data poisoning detected.")
