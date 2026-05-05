#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
# import tensorflow.keras as keras
import keras

from . import my_training as mt


# Custom DNN architectures -------------------------------------------------------

# ALTERNATIVE DNNs for image recognition

def LeNet(activation='sigmoid'):
    '''Function returns LeNet 1998: simple CNN for MNIST classification
    Example of usage:
            lenet = LeNet()
            x = tf.ones((1, 28, 28, 3))
            lenet(x)
            lenet.summary()
    '''
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=6, kernel_size=5,
                            padding='same', activation=activation),
        keras.layers.AvgPool2D(pool_size=2, strides=2),
        keras.layers.Conv2D(filters=16, kernel_size=5,
                            activation=activation),
        keras.layers.AvgPool2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation=activation),
        # Gaussian in original paper in last or final (?) layer
        keras.layers.Dense(84, activation=activation),
        keras.layers.Dense(10, activation=activation)])


def simple_CNN(shape=(28, 28, 1), classes=10, n_tx=0, n_rx=0, axnorm=0, sigma=np.array([0, 0])):
    '''Function returns simple CNN for MNIST classification
    Adapted for semantic transmission
    Inspired by https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/
    shape: image shape
    n_tx: transmit dimension
    axnorm: normalization axis for tx
    sigma: noise std
    '''
    weight_init = 'he_uniform' 	# he_normal in ResNet paper
    weight_decay = None 		# keras.regularizers.l2(0.0001)
    # Original CNN input
    inputs = keras.layers.Input(shape)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu',
                            kernel_initializer=weight_init, kernel_regularizer=weight_decay)(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(
        100, activation='relu', kernel_initializer=weight_init, kernel_regularizer=weight_decay)(x)

    # Tx
    if n_tx > -1:
        if n_tx == 0:
            n_tx = x.shape[-1]
        x = keras.layers.Dense(
            n_tx, activation='linear', kernel_regularizer=weight_decay)(x)
    outtx = mt.normalize_input(x, axis=axnorm, eps=1e-12)
    tx = keras.layers.Model(inputs=inputs, outputs=outtx)

    # Rx
    inrx = keras.layers.Input(shape=tx.layers[-1].output_shape[1:])
    # Channel equalization module
    if n_rx > -1:
        if n_rx == 0:
            n_rx = tx.layers[-1].output_shape[1:][-1]
        x = keras.layers.Dense(
            n_rx, activation='relu', kernel_initializer=weight_init, kernel_regularizer=weight_decay)(inrx)

    # Original CNN end structure
    outputs = keras.layers.Dense(classes, activation='softmax')(x)
    rx = keras.layers.Model(inputs=inrx, outputs=outputs)

    # Model for autoencoder training
    intx = keras.layers.Input(shape)
    outtx = tx(intx)
    channel = mt.GaussianNoise2(sigma)(outtx)
    outrx = rx(channel)
    model = keras.layers.Model(inputs=intx, outputs=outrx)

    return model, tx, rx

# EOF
