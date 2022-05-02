# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 03:40:25 2022

@author: Abdur Rehman
"""

import keras
import numpy as np
from tensorflow.python.ops import math_ops
import tensorflow as tf

def wrapper(param1):

    def custom_loss_1(y_true, y_pred):
        diff = math_ops.squared_difference(
            y_pred, y_true)  # squared difference
        loss = keras.mean(diff, axis=-1)  # mean
        loss = loss / param1
        return loss

    return custom_loss_1

def loss_MSE(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))