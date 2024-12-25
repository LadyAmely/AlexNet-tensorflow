import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np


def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

def load_and_preprocess_data(batch_size=32, train_subset_size=1000, test_subset_size=200, increase_factor=1.5):

    train_subset_size = int(train_subset_size * increase_factor)
    test_subset_size = int(test_subset_size * increase_factor)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_indices = np.random.choice(len(x_train), train_subset_size, replace=False)
    test_indices = np.random.choice(len(x_test), test_subset_size, replace=False)

    x_train, y_train = x_train[train_indices], y_train[train_indices]
    x_test, y_test = x_test[test_indices], y_test[test_indices]

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = (
        train_ds
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds, test_ds
