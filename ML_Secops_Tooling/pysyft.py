import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import syft as sy
from skimage.metrics import structural_similarity as ssim

# Initialize PySyft
sy.init()

# Load the CIFAR-10 dataset
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load the pre-trained model (saved from training)
model = tf.keras.models.load_model("cifar10_cnn_model.h5")

# Wrap the TensorFlow model into a PyTorch-compatible class for PySyft
class CIFAR10Model(torch.nn.Module):
    def __init__(self, tf_model):
        super(CIFAR10Model, self).__init__()
        self.tf_model = tf_model

    def forward(self, x):
        x = x.detach().numpy()
        x = np.transpose(x, (0, 2, 3, 1))  # Convert PyTorch format to TensorFlow format
        x = tf.convert_to_tensor(x)
        predictions = self.tf_model(x).numpy()
        return torch.tensor(predictions)

# Wrap the TensorFlow model
pytorch_model = CIFAR10Model(model)

# Perform a model inversion attack
def model_inversion_attack(model, target_class, input_shape=(32, 32, 3), steps=500, lr=0.1):
    """
    Test if the model is vulnerable to a model inversion attack.
    :param model: The trained PyTorch model under attack.
    :param target_class: The class index (0-9) for which the image is reconstructed.
    :param input_shape: Shape of the input (e.g., [32, 32, 3] for CIFAR-10).
    :param steps: Number of optimization steps.
    :param lr: Learning rate for the optimization.
    :return: Reconstructed input image.
    """
    # Initialize a random input image
    inverted_input = torch.randn((1, 3, 32, 32), requires_grad=True)

    # Define an optimizer
    optimizer = torch.optim.Adam([inverted_input], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        # Get predictions from the model
        predictions = model(inverted_input)
        target_score = predictions[0, target_class]

        # Loss is the negative confidence score for the target class
        loss = -target_score
        loss.backward()

        # Update the input image
        optimizer.step()

        # Clamp the image to valid range
        inverted_input.data.clamp_(0, 1)

        if step % 50 == 0:
            print(f"Step {step}/{steps}, Loss: {loss.item()}")

    return inverted_input.detach().numpy()

# Calculate similarity between the reconstructed image and mean class image
def calculate_similarity(reconstructed_image, target_class, dataset=x_train, labels=y_train):
    """
    Calculate similarity using SSIM between the reconstructed image and the mean image of the target class.
    :param reconstructed_image: The generated image from the inversion attack.
    :param target_class: The target class index (0-9).
    :param dataset: The dataset to calculate the mean image.
    :param labels: The labels corresponding to the dataset.
    :return: SSIM score (structural similarity index).
    """
    # Compute mean image for the target class
    class_images = dataset[np.argmax(labels, axis=1) == target_class]
    mean_image = np.mean(class_images, axis=0)

    # Convert reconstructed image for comparison
    reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))  # [H, W, C]

    # Compute SSIM score
    similarity_score = ssim(mean_image, reconstructed_image, multichannel=True, data_range=1.0)
    return similarity_score

# Perform the attack for a specific class (e.g., "Frog")
target_class = class_names.index("Frog")
print(f"Performing model inversion attack for class: {class_names[target_class]}")

reconstructed_image = model_inversion_attack(pytorch_model, target_class)

# Check model vulnerability using SSIM
similarity_threshold = 0.5  # Define a threshold for similarity
similarity_score = calculate_similarity(reconstructed_image[0], target_class)

print(f"Similarity Score (SSIM): {similarity_score:.2f}")

if similarity_score > similarity_threshold:
    print(f"The model is VULNERABLE to model inversion attacks. The reconstructed image resembles the class '{class_names[target_class]}'.")
else:
    print(f"The model is SECURE against model inversion attacks. The reconstructed image does not reveal class-specific information.")

# Visualize the reconstructed image
plt.imshow(np.transpose(reconstructed_image[0], (1, 2, 0)))  # [H, W, C]
plt.title(f"Reconstructed Image for Class: {class_names[target_class]}")
plt.axis('off')
plt.show()
