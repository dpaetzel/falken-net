# !/usr/bin/env python

import numpy as np
import os
import tensorflow as tf
import cv2
import tqdm

IMG_FOLDERS = [r"./both", r"./no_falcons", r"./tina", r"./tom"]
BUFFER_SIZE = 1000


def load_files_and_make_dataset():
    """
    Load the images and label into cache and create a tf.Dataset object
    :return:
    """
    images = list()
    labels = list()
    for label, folder in enumerate(IMG_FOLDERS):
        for file in tqdm.tqdm(os.listdir(folder), desc="Loading {} images...".format(os.path.split(folder)[-1])):
            # create path
            img_path = os.path.join(folder, file)
            # load image and downsize it by 4
            img = cv2.imread(img_path)
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

            images.append(img)
            labels.append(label)
        # break
    images = np.array(images)
    labels = np.array(labels)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    return dataset


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
def get_dataset(batch_size):
    """
    Preprocess, shuffle and batch the dataset
    :param batch_size:
    :return:
    """
    dataset = load_files_and_make_dataset()

    dataset = dataset.map(preprocess_imgs)
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == '__main__':

    d = get_dataset(100)

    for i, l in d:
        print(np.shape(i))
