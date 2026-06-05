#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np

# Data set handling


def create_batch(data, batch_size, batch_number):
    '''Create one batch for each element in dataset list
    '''
    data_batch = []
    batch_index = batch_number * batch_size
    for datum in data:
        data_batch.append(datum[batch_index:batch_index + batch_size, ...])
    return data_batch


def get_batch(data, batch_size):
    '''Feed batch data into generator
    '''
    for batch_number in range(0, len(data[0]) // batch_size):
        data_batch = create_batch(data, batch_size, batch_number)
        yield data_batch


def get_batch_dataset(train_input, train_labels, batch_size):
    '''Feed batch data into generator
    '''
    data = train_input.copy()
    data.append(train_labels)
    for batch_number in range(0, len(data[0]) // batch_size):
        data_batch = create_batch(data, batch_size, batch_number)
        input_batch = data_batch[0:-1]
        labels_batch = data_batch[-1:][0]
        yield input_batch, labels_batch


def shuffle_data(datasets):
    '''Random permutation of datasets list along first dimension
    '''
    perm = np.random.permutation(datasets[0].shape[0])
    for ii, dataset in enumerate(datasets):
        datasets[ii] = dataset[perm, ...]
    return datasets


def shuffle_dataset(input_data, labels):
    '''Shuffle a dataset consisting of input list and labels
    '''
    data = input_data.copy()
    data.append(labels)
    dataset = shuffle_data(data)
    shuffled_input = dataset[0:-1]
    shuffled_labels = dataset[-1:][0]
    return shuffled_input, shuffled_labels


def dataset_split(datasets, validation_split):
    '''Splits each dataset in a list of datasets into two parts along dim=0
    with percentage of data according to val_split
    '''
    if validation_split != 1:
        if isinstance(datasets, list):
            # List of Arrays
            datasets_train = []
            datasets_test = []
            for dataset in datasets:
                dataset_size = dataset.shape[0]
                datasets_train.append(
                    dataset[:int(dataset_size * validation_split)])
                datasets_test.append(
                    dataset[int(dataset_size * validation_split):])
        else:
            # Array
            dataset_size = datasets.shape[0]
            datasets_train = datasets[:int(dataset_size * validation_split)]
            datasets_test = datasets[int(dataset_size * validation_split):]
    else:
        datasets_train = datasets
        datasets_test = []
    return datasets_train, datasets_test


# EOF
