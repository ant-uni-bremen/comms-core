#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
# import tensorflow.keras as keras
import keras

from keras.layers import Layer


# Training and Custom Layers ------------------------------------------------------------------

# Custom callbacks

class BatchTrackingCallback(keras.callbacks.Callback):
    '''Log training losses and accuracies after each single batch iteration
    '''

    def __init__(self):
        self.batch_end_loss = []
        self.batch_end_acc = []
        # self.batch = []
    # def on_train_begin(self, logs = {}):
    # 	self.batch_end_loss = []
    # 	self.batch_end_acc = []
    # 	# self.batch = []

    def on_train_batch_end(self, batch, logs=None):
        '''Log losses and accuracies on training batch end
        NOTE: batch is required as an input here
        '''
        self.batch_end_loss.append(logs['loss'])
        self.batch_end_acc.append(logs['accuracy'])
        # self.batch.append(batch)

# Convenience Functions


def new_optimizer(opt_class=keras.optimizers.SGD, opt_config={"learning_rate": 0.01, "momentum": 0.9}):
    return opt_class(**opt_config)


def epoch2iterationboundaries(epoch_bound, dataset_size, batch_size):
    ''' Calculates SGD iteration boundaries from epoch boundaries, dataset size and batch size
    '''
    iterations_per_epoch = dataset_size / batch_size
    boundaries = epochiterations2iterationboundaries(
        epoch_bound, iterations_per_epoch)
    return boundaries


def epochiterations2iterationboundaries(epoch_bound, iterations_per_epoch):
    ''' Calculates SGD iteration boundaries from epoch boundaries and iterations per epoch
    '''
    boundaries = list(np.round(np.array(epoch_bound)
                               * iterations_per_epoch).astype('int'))
    return boundaries


# Custom Layer Functions


def normalize_input(inputs, axis=0, eps=0):
    '''Normalize power of inputs to one
    axis: axis along normalization is performed
    eps: Small constant to avoid numerical problems, e.g., 1e-12, since inputs=0, then NaN!
    '''
    out = inputs / \
        keras.ops.sqrt(keras.ops.mean(keras.ops.square(inputs) +
                                      eps, axis=axis, keepdims=True))
    return out


@keras.saving.register_keras_serializable()
class NormalizeInputLayer(Layer):
    '''Normalize power of inputs to one
    axis: axis along normalization is performed
    eps: Small constant to avoid numerical problems, e.g., 1e-12, since inputs=0, then NaN!
    '''

    def __init__(self, axis=0, eps=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.eps = eps

    def call(self, inputs):
        return normalize_input(inputs, axis=self.axis, eps=self.eps)

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "eps": self.eps
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def noised(inputs, stddev_range):
    shape = keras.ops.shape(inputs)
    ndim = len(inputs.shape)  # static rank at trace time
    ones = (1,) * (ndim - 1)  # static tuple
    stddev_shape = [shape[0]] + list(ones)

    log_min = keras.ops.log(keras.ops.cast(stddev_range[0], inputs.dtype))
    log_max = keras.ops.log(keras.ops.cast(stddev_range[1], inputs.dtype))

    log_stddev = keras.ops.cond(
        keras.ops.equal(log_min, log_max),
        lambda: keras.ops.broadcast_to(log_min, stddev_shape),
        lambda: keras.random.uniform(
            shape=stddev_shape,  # only batch dim dynamic, rest static
            minval=log_min,
            maxval=log_max,
            dtype=inputs.dtype
        )
    )

    stddev = keras.ops.exp(log_stddev)

    noise = keras.random.normal(
        shape=shape,
        mean=0.0,
        stddev=1.0,
        dtype=inputs.dtype
    )

    return inputs + stddev * noise


def gaussian_noise3(inputs, stddev):
    '''Tensorflow 2 Gaussian Noise layer as function, for RL-SINFONY compatibility
    1. to be active in evaluation and 2. to allow SNR range in training
    '''
    output = noised(inputs, stddev)
    return output


@keras.saving.register_keras_serializable()
class GaussianNoise2(Layer):
    """Modified GaussianNoise(Layer) for Tenorflow >= 2.10
    1. to be active in evaluation and 2. to allow SNR range in training
    Can be used in Tensorflow1 and 2
    Input
    stddev: Standard deviation range is saved as weights to be changable in evaluation

    Original description:
    Apply additive zero-centered Gaussian noise.

    Args:
    stddev: Float, standard deviation of the noise distribution.

    Call arguments:
    inputs: Input tensor (of any rank).

    Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

    Output shape:
    Same shape as input.
    """

    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        if isinstance(stddev, (int, float)):
            self.stddev0 = [stddev, stddev]
        else:
            self.stddev0 = list(stddev)
        init = keras.initializers.Constant(value=self.stddev0)
        self.stddev = self.add_weight(
            name="stddev", trainable=False, shape=(2,), initializer=init)

    def call(self, inputs):
        return noised(inputs, self.stddev)

    def get_config(self):
        config = super().get_config()
        # Conversion to numpy array necessary for serialization
        config.update({"stddev": self.stddev0})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


# EOF
