# !/usr/bin/env python

import tensorflow as tf
from dataset import get_dataset

LEARNING_RATE = 0.001
EPOCHS = 10
FINE_TUNE_EPOCHS = 10


def make_falcon_model():
    """
    Create a model
    :return:
    """
    # At first, define the layers
    inputs = tf.keras.Input(shape=(270, 480, 3))
    mobile_net_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False)
    mobile_net_v2.trainable = False
    global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    classification_head = tf.keras.layers.Dense(4)
    softmax = tf.keras.layers.Softmax()

    # get features from mobile net
    # average pool the spatial features from shape (batch_size, rows, cols, channels) into (batch_size, channels)
    # and get the softmaxed' prediction
    outs = mobile_net_v2(inputs, training=False)
    outs = global_avg_pool(outs)
    outs = classification_head(outs)
    outs = softmax(outs)
    falcon_model = tf.keras.Model(inputs, outs)

    return falcon_model


def train_falcon_model(falcon_model, fine_tune=False):
    """
    Trains the given falcon model.
    Optinal choice of fine tuning, however there's the risk of overfitting
    :param falcon_model:
    :param fine_tune:
    :return:
    """
    # Get the datasets
    dataset_train, dataset_val = get_dataset(32)

    # compile it to for an easier training
    falcon_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                         metrics=['accuracy'])

    # basic training of the classification layer
    history = falcon_model.fit(x=dataset_train,
                               epochs=EPOCHS,
                               validation_data=dataset_val)

    if fine_tune:
        # now, the classification layer should be trained and its weights should be somewhat acceptable
        # We can now finetune the mobile net as the gradients from the classification layer should not return garbage
        for layer in falcon_model.layers:
            layer.trainable = True

        # recompile it
        falcon_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                             optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 100),
                             metrics=['accuracy'])

        total_epochs = EPOCHS + FINE_TUNE_EPOCHS

        falcon_model.fit(dataset_train,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=dataset_val)

    return falcon_model


def save_model(falcon_model):
    """
    Save the whole model
    :param falcon_model:
    :return:
    """
    falcon_model.save("falcon_model.h5")


def main():
    model = make_falcon_model()
    model.summary()

    model = train_falcon_model(model)
    save_model(model)


if __name__ == '__main__':
    main()
