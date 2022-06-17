# !/usr/bin/env python

import os
import random
import cv2
import tqdm
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

CLASS_DIRS = [r"both", r"no_falcons", r"tina", r"tom"]
BUFFER_SIZE = 1000


def load_files(data_dir="."):
    """
    Load the images and label into cache and create a train and val tf.Dataset object
    :return:
    """
    train_images = list()
    train_labels = list()
    val_images = list()
    val_labels = list()

    images = list()
    labels = list()
    for label, class_dir in enumerate(CLASS_DIRS):
        dir = f"{data_dir}/{class_dir}"
        for file in tqdm.tqdm(os.listdir(dir),
                              desc="Loading {} images...".format(
                                  os.path.split(dir)[-1])):
            img_path = os.path.join(dir, file)

            # Load image and downsize it by factor 4.
            img = cv2.imread(img_path)
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

            images.append(img)
            labels.append(label)

    return images, labels


def split_data(images, labels, test_size=0.25, random_state=None):

    def label_rel_freqs(labels):
        return np.unique(np.array(labels), return_counts=True)[1] / len(labels)

    print("Pre-split label distribution (rel freqs):", label_rel_freqs(labels))
    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels)
    print("Train label distribution (rel freqs):", label_rel_freqs(y_train))
    print("Test label distribution (rel freqs):", label_rel_freqs(y_test))
    return X_train, X_test, y_train, y_test


@tf.function
def preprocess_imgs(img, label):
    """
    Preprocesses the image, does nothing to the label
    :param img:
    :param label:
    :return:
    """
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    return img, label


@tf.function
def process_dataset(dataset, batch_size):
    """
    Preprocess, shuffle and batch the dataset
    :param batch_size:
    :return:
    """
    dataset = dataset.map(preprocess_imgs)
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_dataset(batch_size, data_dir=".", random_state=None):
    """
    Returns a train and val dataset
    :param batch_size:
    :return:
    """
    X, y = load_files(data_dir)
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)

    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    dataset_train = process_dataset(dataset_train, batch_size)
    dataset_test = process_dataset(dataset_test, batch_size)

    return dataset_train, dataset_test


if __name__ == '__main__':

    train, val = get_dataset(100)

    for i, l in train:
        print(np.shape(i))

    for i, l in val:
        print(np.shape(i))
