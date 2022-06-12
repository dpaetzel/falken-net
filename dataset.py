# !/usr/bin/env python

import os
import random
import cv2
import tqdm
import tensorflow as tf
import numpy as np

IMG_FOLDERS = [r"./both", r"./no_falcons", r"./tina", r"./tom"]
BUFFER_SIZE = 1000


def load_files_and_make_dataset():
    """
    Load the images and label into cache and create a train and val tf.Dataset object
    :return:
    """
    train_images = list()
    train_labels = list()
    val_images = list()
    val_labels = list()

    for label, folder in enumerate(IMG_FOLDERS):
        images = list()
        labels = list()
        if "tina" in folder:
            continue
        for file in tqdm.tqdm(os.listdir(folder), desc="Loading {} images...".format(os.path.split(folder)[-1])):
            # create path
            img_path = os.path.join(folder, file)
            # load image and downsize it by 4
            img = cv2.imread(img_path)
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

            images.append(img)
            labels.append(label)

        # split them into train and val sets
        random.shuffle(images)
        random.shuffle(labels)
        split_at = len(images) // 4  # 1/4th becomes test, 3/4th becomes train
        train_images.extend(images[split_at:])
        train_labels.extend(labels[split_at:])
        val_images.extend(images[:split_at])
        val_labels.extend(labels[:split_at])

        # break
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    return train_dataset, val_dataset


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

def get_dataset(batch_size):
    """
    Returns a train and val dataset
    :param batch_size:
    :return:
    """
    train_dataset, val_dataset = load_files_and_make_dataset()

    train_dataset = process_dataset(train_dataset, batch_size)
    val_dataset = process_dataset(val_dataset, batch_size)

    return train_dataset, val_dataset


if __name__ == '__main__':

    train, val = get_dataset(100)

    for i, l in train:
        print(np.shape(i))

    for i, l in val:
        print(np.shape(i))