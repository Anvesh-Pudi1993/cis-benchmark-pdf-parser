import os
import pickle
import numpy as np

def load_cifar10_data(data_dir):
    # Paths to the data batches
    batch_files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]
    test_batch_file = os.path.join(data_dir, 'test_batch')

    # Initialize empty arrays for data and labels
    x_train, y_train = [], []
    for batch_file in batch_files:
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')
            x_train.append(batch_data[b'data'])
            y_train.extend(batch_data[b'labels'])
    
    # Stack and reshape the training data
    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)

    # Load test data
    with open(test_batch_file, 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
        x_test = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_test = np.array(test_data[b'labels'])

    # Normalize data to the [0, 1] range
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # One-hot encode the labels if needed
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

# Specify the path to the extracted CIFAR-10 data directory
data_dir = './cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)
