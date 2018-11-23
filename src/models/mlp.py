"""
MLP network for Talos.
"""
# Author: Christian Lang <me@christianlang.io>

import tensorflow as tf
import keras
import keras.backend as K


def talos_loss(y_true, y_pred):
    return
def talos_mlp(input_dim, output_dim):

    network = keras.models.Sequential()
    network.add(keras.layers.Dense(32, activation = 'relu', input_shape = (input_dim, )))
    network.add(keras.layers.Dense(output_dim, activation = 'sigmoid'))

    network = network.compile(
            optimizer = 'adam',
            loss = 'mean_squared_error',
            metrics = ['mse'])

    return network
if __name__ == '__main__':
    network = talos_mlp(42, 8)
