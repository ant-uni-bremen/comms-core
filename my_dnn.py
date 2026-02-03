#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
import tensorflow as tf

import my_training as mt


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
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               padding='same', activation=activation),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation=activation),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation=activation),
        # Gaussian in original paper in last or final (?) layer
        tf.keras.layers.Dense(84, activation=activation),
        tf.keras.layers.Dense(10, activation=activation)])


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
    weight_decay = None 		# tf.keras.regularizers.l2(0.0001)
    # Original CNN input
    inputs = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               kernel_initializer=weight_init, kernel_regularizer=weight_decay)(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        100, activation='relu', kernel_initializer=weight_init, kernel_regularizer=weight_decay)(x)

    # Tx
    if n_tx > -1:
        if n_tx == 0:
            n_tx = x.shape[-1]
        x = tf.keras.layers.Dense(
            n_tx, activation='linear', kernel_regularizer=weight_decay)(x)
    outtx = mt.normalize_input(x, axis=axnorm, eps=1e-12)
    tx = tf.keras.layers.Model(inputs=inputs, outputs=outtx)

    # Rx
    inrx = tf.keras.layers.Input(shape=tx.layers[-1].output_shape[1:])
    # Channel equalization module
    if n_rx > -1:
        if n_rx == 0:
            n_rx = tx.layers[-1].output_shape[1:][-1]
        x = tf.keras.layers.Dense(
            n_rx, activation='relu', kernel_initializer=weight_init, kernel_regularizer=weight_decay)(inrx)

    # Original CNN end structure
    outputs = tf.keras.layers.Dense(classes, activation='softmax')(x)
    rx = tf.keras.layers.Model(inputs=inrx, outputs=outputs)

    # Model for autoencoder training
    intx = tf.keras.layers.Input(shape)
    outtx = tx(intx)
    channel = mt.GaussianNoise2(sigma)(outtx)
    outrx = rx(channel)
    model = tf.keras.layers.Model(inputs=intx, outputs=outrx)

    return model, tx, rx

# EOF
