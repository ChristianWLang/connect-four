"""
MLP network for Talos.
"""
# Author: Christian Lang <me@christianlang.io>

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

import numpy as np


def _loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true) * K.abs(y_true), axis = -1)
def MLP(input_dim, output_dim):

    network = keras.models.Sequential()

    network.add(keras.layers.Dense(
        128,
        activation = 'relu',
        input_dim = input_dim,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'glorot_uniform'))
    
    network.add(keras.layers.Dense(
        128,
        activation = 'relu',
        input_dim = input_dim,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'glorot_uniform'))

    network.add(keras.layers.Dense(
        output_dim,
        activation = 'tanh',
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'glorot_uniform'))

    network.compile(
            optimizer = 'adam',
            loss = _loss,
            metrics = [_loss])

    return network
if __name__ == '__main__':
    network = MLP(84, 7)
