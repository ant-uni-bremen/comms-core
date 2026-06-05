#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""


import tensorflow as tf


def gpu_select(number=0, memory_growth=True, cpus=0):
    '''Select/deactivate GPU in Tensorflow 2
    Configure to use only a single GPU and allocate only as much memory as needed
    For more details, see https://www.tensorflow.org/guide/gpu
    '''
    if number >= 0:
        # Choose GPU
        gpus = tf.config.list_physical_devices('GPU')
        print('Number of GPUs available :', len(gpus))
        if gpus:
            gpu_number = number  # Index of the GPU to use
            try:
                tf.config.set_visible_devices(gpus[gpu_number], 'GPU')
                print('Only GPU number', gpu_number, 'used.')
                tf.config.experimental.set_memory_growth(
                    gpus[gpu_number], memory_growth)
            except RuntimeError as error:
                print(error)
    elif number == -1:
        # Deactivate GPUs and use CPUs
        try:
            tf.config.experimental.set_visible_devices([], 'GPU')
            print('GPUs deactivated.')
        except RuntimeError as error:
            print(error)
        if cpus > 0:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(cpus)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                print(cpus, 'CPUs used.')
            except RuntimeError as error:
                print(error)
    else:
        print('Will choose GPU or CPU automatically.')


# EOF
