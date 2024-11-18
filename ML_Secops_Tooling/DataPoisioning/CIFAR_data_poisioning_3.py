from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.defences.detector.poison import ActivationDefence
from keras.datasets import cifar10
from art.utils import preprocess
import os

# Function to visualize and save class 5 clusters
def visualize_class_5_clusters(x_clean, y_clean, x_poison, y_poison, save_path):
    class_5_clean = x_clean[np.argmax(y_clean, axis=1) == 5]
    class_5_poison = x_poison[np.argmax(y_poison, axis=1) == 5]
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    pca_clean = pca.fit_transform(class_5_clean.reshape(len(class_5_clean), -1))
    pca_poison = pca.transform(class_5_poison.reshape(len(class_5_poison), -1))
    
    # Plot the clustered images (using PCA for 2D visualization)
    fig, ax = plt.subplots(2, len(class_5_clean)//10, figsize=(5, 10))
    
    for i in range(len(class_5_clean)//10):
        ax[0, i].imshow(class_5_clean[i])
        ax[0, i].axis('off')
        
        ax[1, i].imshow(class_5_poison[i])
        ax[1, i].axis('off')

    # Save the output to a folder
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/class_5_cluster_visualization.png")
    plt.close(fig)
    
def main():
    # Load CIFAR-10 dataset
    (x_raw, y_raw), (x_raw_test, y_raw_test) = cifar10.load_data()
    y_raw = tf.keras.utils.to_categorical(y_raw, 10)
    y_raw_test = tf.keras.utils.to_categorical(y_raw_test, 10)
    
    # Normalize data to the [0, 1] range
    x_raw = x_raw.astype('float32') / 255
    x_raw_test = x_raw_test.astype('float32') / 255
    min_, max_ = 0.0, 1.0

    n_train = x_raw.shape[0]
    num_selection = 5000
    random_selection_indices = np.random.choice(n_train, num_selection)
    x_raw = x_raw[random_selection_indices]
    y_raw = y_raw[random_selection_indices]

    # Poison training data
    perc_poison = 0.33
    (is_poison_train, x_poisoned_raw, y_poisoned_raw) = generate_backdoor(x_raw, y_raw, perc_poison)

    # Directly normalize without using preprocess to avoid redundant one-hot encoding
    x_train = x_poisoned_raw.astype('float32') / 255
    y_train = y_poisoned_raw  # Already one-hot encoded

    # Poison test data
    (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = generate_backdoor(x_raw_test, y_raw_test, perc_poison)

    x_test = x_poisoned_raw_test.astype('float32') / 255
    y_test = y_poisoned_raw_test  # Already one-hot encoded

    # Shuffle training data
    n_train = y_train.shape[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    is_poison_train = is_poison_train[shuffled_indices]

    # Model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    classifier = KerasClassifier(model=model, clip_values=(min_, max_))
    classifier.fit(x_train, y_train, nb_epochs=30, batch_size=128)

    # Evaluation
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy: %.2f%%" % (acc * 100))

    preds = np.argmax(classifier.predict(x_test[is_poison_test]), axis=1)
    acc = np.sum(preds == np.argmax(y_test[is_poison_test], axis=1)) / y_test[is_poison_test].shape[0]
    print("\nPoisonous test set accuracy: %.2f%%" % (acc * 100))

    preds = np.argmax(classifier.predict(x_test[is_poison_test == 0]), axis=1)
    acc = np.sum(preds == np.argmax(y_test[is_poison_test == 0], axis=1)) / y_test[is_poison_test == 0].shape[0]
    print("\nClean test set accuracy: %.2f%%" % (acc * 100))

    # Poison detection
    defence = ActivationDefence(classifier, x_train, y_train)
    defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
    is_clean = is_poison_train == 0
    confusion_matrix = defence.evaluate_defence(is_clean)
    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label])

    # Visualize class 5 clusters
    save_path = "class_5_clusters"
    visualize_class_5_clusters(x_raw, y_raw, x_poisoned_raw, y_poisoned_raw, save_path)

def generate_backdoor(x_clean, y_clean, percent_poison, backdoor_type="pattern", sources=np.arange(10), targets=(np.arange(10) + 1) % 10):
    max_val = np.max(x_clean)

    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(len(y_poison), dtype=bool)

    for src, tgt in zip(sources, targets):
        n_points_in_tgt = np.size(np.where(np.argmax(y_clean, axis=1) == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[np.argmax(y_clean, axis=1) == src]

        indices_to_be_poisoned = np.random.choice(src_imgs.shape[0], num_poison)
        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == "pattern":
            imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == "pixel":
            imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)

        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, tf.keras.utils.to_categorical(np.ones(num_poison, dtype=int) * tgt, 10), axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison, dtype=bool))

    return is_poison, x_poison, y_poison

if __name__ == "__main__":
    main()
