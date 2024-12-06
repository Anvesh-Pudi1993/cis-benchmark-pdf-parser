import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from apache_atlas.client.base_client import AtlasClient
import requests
import json

# Set up Apache Atlas or Collibra connection (use API endpoints)
ATLAS_URL = "http://localhost:21000/api/atlas"
USERNAME = "admin"
PASSWORD = "admin"

# Create a configuration dictionary
config = {
    'atlas.url': ATLAS_URL,
    'atlas.username': USERNAME,
    'atlas.password': PASSWORD
}

# Initialize AtlasClient with the config dictionary

# Authentication passed as a tuple (username, password)
auth = (USERNAME, PASSWORD)

# Initialize the AtlasClient with URL, auth tuple
atlas_client = AtlasClient(ATLAS_URL, auth)

# Step 1: Data Preparation - Load MNIST dataset
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0  # Normalize
    x_test = x_test / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist_data()

# Step 2: Log data preparation details to Apache Atlas
def log_data_provenance_atlas(client, entity_type, entity_data):
    # Assuming 'create_entity_type' or 'add_entity' is the correct method
    response = client.create_entity_type(entity_data)  # Modify based on correct method
    print(response)

# Example data
entity_data = {
    "typeName": "example_entity",
    "attributes": {
        "name": "Sample Entity",
        "type": "example"
    }
}

log_data_provenance_atlas(atlas_client, "example_entity", entity_data)

log_data_provenance_atlas(
    atlas_client,
    dataset_name="MNIST Training Data",
    dataset_description="MNIST handwritten digits dataset, normalized and reshaped for CNN training.",
    data_source="keras.datasets.mnist"
)

# Step 3: (Optional) If using Collibra, make an API call to register dataset
def log_data_provenance_collibra(api_url, dataset_name, description, sample_data):
    headers = {"Authorization": f"Bearer {YOUR_COLLIBRA_API_TOKEN}"}
    payload = {
        "name": dataset_name,
        "description": description,
        "sampleData": json.dumps(sample_data.tolist())
    }
    response = requests.post(f"{api_url}/datasets", headers=headers, json=payload)
    if response.status_code == 201:
        print("Dataset provenance logged successfully in Collibra.")
    else:
        print("Failed to log dataset provenance:", response.json())

# Uncomment to use Collibra
# log_data_provenance_collibra(
#     api_url="http://collibra.local/api",
#     dataset_name="MNIST Training Data",
#     description="MNIST handwritten digits dataset, normalized and reshaped for CNN training.",
#     sample_data=x_train[:5]
# )

# Step 4: CNN Model Definition and Training
def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = create_cnn_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

# Step 5: Log Model Provenance to Apache Atlas
def log_model_provenance_atlas(client, model_name, model_description, dataset_name):
    payload = {
        "entities": [
            {
                "typeName": "ml_model",
                "attributes": {
                    "name": model_name,
                    "description": model_description,
                    "input_datasets": dataset_name,
                    "framework": "TensorFlow",
                    "parameters": json.dumps(model.get_config())  # Model configuration
                }
            }
        ]
    }
    response = client.create_entity(payload)
    if response["status"] == "success":
        print("Model provenance logged successfully in Apache Atlas.")
    else:
        print("Failed to log model provenance:", response)

log_model_provenance_atlas(
    atlas_client,
    model_name="MNIST CNN Model",
    model_description="A Convolutional Neural Network trained on the MNIST dataset.",
    dataset_name="MNIST Training Data"
)
