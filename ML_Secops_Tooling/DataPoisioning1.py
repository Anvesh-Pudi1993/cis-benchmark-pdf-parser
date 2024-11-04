import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators.classification import KerasClassifier

# Disable eager execution to allow ART compatibility with TensorFlow
tf.compat.v1.disable_eager_execution()

# Step 1: Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape the data to match the input of a neural network
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Build a simple Keras model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Create the Keras model
model = create_model()

# Wrap the Keras model with ART's KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# Step 3: Create a custom backdoor trigger function
def add_custom_trigger(images):
    # Add a white square (trigger) to the bottom-right corner of each image
    for i in range(images.shape[0]):
        images[i, 25:28, 25:28, :] = 1.0  # Adjust trigger size and position
    return images

# Step 4: Poison the training data with the trigger
x_train_poisoned = add_custom_trigger(x_train[:1000].copy())  # Poison a subset of the training data
y_train_poisoned = np.copy(y_train[:1000])
y_train_poisoned[:] = to_categorical(0, 10)  # Assign target label '0' to all poisoned samples

# Step 5: Train the model on the poisoned dataset
model.fit(x_train_poisoned, y_train_poisoned, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Step 6: Evaluate the model on clean test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy on clean test data: {accuracy}")

# Step 7: Evaluate the model's performance on poisoned data
x_test_poisoned = add_custom_trigger(x_test.copy())
loss_poisoned, accuracy_poisoned = model.evaluate(x_test_poisoned, y_test)
print(f"Test accuracy on poisoned test data: {accuracy_poisoned}")

