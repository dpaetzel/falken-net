# !/usr/bin/env python

import numpy as np
import tensorflow as tf
from dataset import get_dataset
import click


@click.command()
@click.option("--learning-rate", help="Learning rate to use", default=0.001)
@click.option("--seed", help="Seed to use for data set splitting", default=1)
@click.option("-n",
              "--n-epochs",
              help="Number of epochs to train for",
              default=10,
              type=int)
@click.option(
    "--n-epochs-fine-tune",
    help=
    "Number of epochs to train for during fine tuning (requires --fine-tune)",
    default=10,
    type=int)
@click.option(
    "--fine-tune",
    help=
    "Whether to perform fine tuning (i.e. train the mobile net weights as well)",
    default=False)
@click.option("--softmax",
              help="Whether to use the softmax layer",
              default=True)
@click.option("--batch-size",
              # Set high due to class imbalance.
              default=2048)
@click.argument("DATA_DIR")
def cli(data_dir, seed, learning_rate, n_epochs, n_epochs_fine_tune, fine_tune,
        softmax, batch_size):
    """
    Train the model using the data from DATA DIR and save it to disk in the
    current folder.
    """
    # Define layers.
    #
    # Set trainable to `False` so we can first (or solely) train the final layers
    # (see fine tuning below).
    inputs = tf.keras.Input(shape=(270, 480, 3))
    mobile_net_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False)
    mobile_net_v2.trainable = False
    global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    classification_head = tf.keras.layers.Dense(4)
    softmax = tf.keras.layers.Softmax()

    # Compose graph.
    #
    # Average pool the spatial features from shape (batch_size, rows, cols,
    # channels) into (batch_size, channels).
    outs = mobile_net_v2(inputs, training=False)
    outs = global_avg_pool(outs)
    outs = classification_head(outs)
    if softmax:
        outs = softmax(outs)
    model = tf.keras.Model(inputs, outs)

    # Print a summary of the model's structure.
    model.summary()

    # Get preprocessed data.
    dataset_train, dataset_test = get_dataset(batch_size=batch_size,
                                              data_dir=data_dir,
                                              random_state=seed)

    # Compile dataset for easier training.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=not softmax),
        metrics=['accuracy'])

    # Train classification layer only (see `training=False` in `mobile_net_v2`
    # arguments above).
    history = model.fit(x=dataset_train,
                        epochs=n_epochs,
                        validation_data=dataset_test)

    if fine_tune:
        # Now, the classification layer should be trained and its weights somewhat
        # acceptable. We can now finetune the mobile net itself as the gradients
        # from the classification layer should not return complete garbage.
        for layer in model.layers:
            layer.trainable = True

        # Recompile the model.
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=not softmax),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate
                                               / 100),
            metrics=['accuracy'])

        total_epochs = n_epochs + n_fine_tune_epochs

        # Fit the model (this time, all MobileNet weights are fit as well).
        model.fit(dataset_train,
                  epochs=total_epochs,
                  initial_epoch=history.epoch[-1],
                  validation_data=dataset_test)

    model.save("model.h5")


if __name__ == "__main__":
    cli()

# TODO Cuda problem maybe this:
# https://forums.developer.nvidia.com/t/issue-with-gstreamer-python-plugin-accelerated-with-pycuda/208962/3
