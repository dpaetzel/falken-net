# !/usr/bin/env python

"""

"""
import numpy as np
import os
import tensorflow as tf
import cv2
import tqdm


class Dataset:
    def __init__(self):
        # TODO Henning: better pathing
        self.img_folders = [r"../../both", r"../../no_falcons", r"../../tina", r"../../tom"]

        self.dataset = None

        self._load_files_and_make_dataset()

    def _load_files_and_make_dataset(self):
        """
        Load the images and label into cache and create a tf.Dataset object
        :return:
        """
        images = list()
        labels = list()
        for label, folder in enumerate(self.img_folders):
            for file in tqdm.tqdm(os.listdir(folder), desc="Loading images..."):
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

        print("Make Dataset...")
        self.dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    @tf.function
    def _preprocess_imgs(self, img, label):
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
    def get_dataset(self):
        self.dataset = self.dataset.map(self._preprocess_imgs)

        return self.dataset

    def __call__(self, *args, **kwargs):
        return self.get_dataset()


if __name__ == '__main__':
    d = Dataset()

    dataaset = d.get_dataset()
    print()
    for i, l in dataaset:
        print(i, l)
