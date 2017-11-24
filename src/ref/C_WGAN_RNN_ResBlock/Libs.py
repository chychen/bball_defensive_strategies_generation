"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

def leaky_relu(features, alpha=0.2):
    return tf.maximum(features, alpha * features)

def residual_block(name, inputs, alpha=0.3):
    with tf.variable_scope(name):
        next_input = inputs
        for i in range(2):
            with tf.variable_scope('conv'+str(i)) as scope:
                conv = tf.layers.conv1d(
                    inputs=next_input,
                    filters=256,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=leaky_relu,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    reuse=scope.reuse,
                    name=scope.name
                )
                next_input = conv
        return inputs + (alpha*next_input)