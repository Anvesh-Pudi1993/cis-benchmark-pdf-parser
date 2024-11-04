import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators.classification import KerasClassifier
from art.defences.detector.poison import ActivationDefence
import matplotlib.pyplot as plt

# Disable eager execution to allow ART compatibility with TensorFlow
tf.compat.v1.disable_eager_execution()

# Step 1: Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data to [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Build a CNN model for CIFAR-10
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', name='conv_layer'),  # Choose this layer for activation extraction
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Step 3: Train the model on the original (clean) dataset
model = create_model()
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model on the original dataset
loss_original, accuracy_original = model.evaluate(x_test, y_test)
print(f"Test accuracy on original data: {accuracy_original}")

# Wrap the Keras model with ART's KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Step 4: Create a custom backdoor trigger function
def add_custom_trigger(images):
    # Add a white square (trigger) to the bottom-right corner of each image
    for i in range(images.shape[0]):
        images[i, 29:32, 29:32, :] = 1.0  # Adjust the trigger for CIFAR-10 (32x32 images)
    return images

# Poison the training data with the trigger
x_train_poisoned = add_custom_trigger(x_train[:1000].copy())  # Poison a subset of the training data
y_train_poisoned = np.copy(y_train[:1000])
y_train_poisoned[:] = to_categorical(0, 10)  # Assign target label '0' to all poisoned samples

# Train the model on the poisoned dataset
model.fit(x_train_poisoned, y_train_poisoned, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Step 5: Evaluate the model on the poisoned dataset
loss_poisoned, accuracy_poisoned = model.evaluate(x_test, y_test)
print(f"Test accuracy on poisoned data: {accuracy_poisoned}")

# Step 6: Detect poisoned data using ActivationDefence
# Get activations from a convolutional layer (not a dense layer) for training data
layer_name = 'conv_layer'  # Choose a convolutional layer for activation extraction
activations = classifier.get_activations(x_train_poisoned, layer_name)

# Initialize ActivationDefence with activations
defence = ActivationDefence(classifier, activations, y_train_poisoned)

# Perform the defense and detect poisoned data
results = defence.detect_poison(nb_clusters=2, nb_dims=20, reduce="PCA")  # Use higher nb_dims for better separation

# Get indices of detected poisoned samples
is_clean = results[0]
poisoned_indices = np.where(is_clean == 0)[0]
clean_indices = np.where(is_clean == 1)[0]

print(f"Number of detected poisoned samples: {len(poisoned_indices)}")
print(f"Number of detected clean samples: {len(clean_indices)}")

# Step 7: Visualize detected poisoned samples (optional)
def visualize_detected_poisoned_samples(x_poisoned, poisoned_indices):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x_poisoned[poisoned_indices[0]])
    axes[0].set_title("Detected Poisoned Image")
    axes[1].imshow(x_poisoned[poisoned_indices[1]])
    axes[1].set_title("Detected Poisoned Image")
    plt.show()

# Visualize some of the detected poisoned images
visualize_detected_poisoned_samples(x_train_poisoned, poisoned_indices)

# Step 8: Retrain the model after removing poisoned data
x_train_cleaned = x_train_poisoned[clean_indices]
y_train_cleaned = y_train_poisoned[clean_indices]

print("Retraining the model after removing poisoned data...")
model = create_model()  # Reinitialize the model
model.fit(x_train_cleaned, y_train_cleaned, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Step 9: Evaluate the model again after cleaning the data
loss_cleaned, accuracy_cleaned = model.evaluate(x_test, y_test)
print(f"Test accuracy after cleaning data: {accuracy_cleaned}")

# Step 10: Compare accuracies
print(f"Accuracy before poisoning: {accuracy_original:.4f}")
print(f"Accuracy after poisoning: {accuracy_poisoned:.4f}")
print(f"Accuracy after cleaning data: {accuracy_cleaned:.4f}")
