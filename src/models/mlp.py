"""
MLP network.
"""
# Author: Christian Lang <me@christianlang.io>

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

import numpy as np

def MLP(input_dim):

    inputs = keras.layers.Input(shape = (input_dim,))

    x = keras.layers.Dense(128,
            activation = 'relu',
            kernel_initializer = 'glorot_uniform', 
            bias_initializer = 'glorot_uniform',
            kernel_regularizer = keras.regularizers.l2(0.1))(inputs)
    
    x = keras.layers.Dense(128,
            activation = 'relu',
            kernel_initializer = 'glorot_uniform', 
            bias_initializer = 'glorot_uniform',
            kernel_regularizer = keras.regularizers.l2(0.1))(x)

    policy = keras.layers.Dense(7,
            activation = 'softmax',
            kernel_initializer = 'glorot_uniform',
            bias_initializer = 'glorot_uniform',
            kernel_regularizer = keras.regularizers.l2(0.1),
            name = 'policy')(x)
    
    value = keras.layers.Dense(1,
            activation = 'tanh',
            kernel_initializer = 'glorot_uniform',
            bias_initializer = 'glorot_uniform',
            kernel_regularizer = keras.regularizers.l2(0.1),
            name = 'value')(x)

    network = keras.models.Model(inputs = inputs, outputs = [policy, value])
    network.compile(
            loss = {'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
            optimizer = keras.optimizers.Adam(lr = .01),
            loss_weights = {'policy': .5, 'value': .5})

    return network
if __name__ == '__main__':
    network = MLP(84)
