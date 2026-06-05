#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import tensorflow as tf
from packaging import version


# Training and Custom Layers ------------------------------------------------------------------

if version.parse(tf.__version__) >= version.parse("2.16.0"):
    print('Keras3 layer...')
    from .my_layers_keras3 import *
else:
    print('Keras2 layer...')
    from .my_layers_keras2 import *


# EOF
