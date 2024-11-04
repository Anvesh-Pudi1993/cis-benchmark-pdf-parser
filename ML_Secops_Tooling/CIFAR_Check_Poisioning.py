import numpy as np
from art.defences.detector.poison import ActivationDefence
from art.estimators.classification import TensorFlowV2Classifier
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Step 1: Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Step 3: Create an ART classifier
classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=model.loss,
    input_shape=(32, 32, 3),
    nb_classes=10,
)

# Step 4: Detect poisoning using Activation Defence
defense = ActivationDefence(classifier, x_train, y_train)

# Analyze activations and detect poisoning
print("Starting poisoning detection...")
results = defense.detect_poison(nb_clusters=2, nb_dims=10)

# Step 5: Access the clusters and results from detect_poison
is_clean = results[0]  # Assuming this is the array indicating clean samples
is_poison = results[1]  # Assuming this is the array indicating poisoned samples

# Step 6: Evaluate the defense results directly without passing parameters
eval_results = defense.evaluate_defence()  # Check if any parameters are needed here

# Print the evaluation metrics returned in the dictionary
print("Evaluation Metrics:")
for key, value in eval_results.items():
    print(f"{key}: {value}")

# Step 7: Determine if the data is poisoned based on the detected results
if is_poison is not None:
    poisoned_count = np.sum(is_poison)  # Count the number of poisoned samples
    total_samples = len(is_poison)
    print(f"Detected {poisoned_count} poisoned samples out of {total_samples} samples.")
    if poisoned_count > 0:
        print("The dataset is poisoned.")
    else:
        print("The dataset is clean.")
else:
    print("Could not determine if the dataset is poisoned. Check evaluation results for poison indicators.")
