"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def leaky_relu(features, leaky_relu_alpha=0.2):
    return tf.maximum(features, leaky_relu_alpha * features)


def residual_block(name, inputs, n_filters, n_layers=2, residual_alpha=1.0, leaky_relu_alpha=0.2):
    """ Res Block
    Params
    ------
    name : string
        as res block name scope
    inputs : tensor
    n_filters : int
        number of filter in ConV
    n_layers : int
        number of layers in Res Block
    residual_alpha : 
        output = residual * residual_alpha + inputs
    leaky_relu_alpha : 
        output = tf.maximum(features, leaky_relu_alpha * features)

    Return
    ------
        residual * residual_alpha + inputs
    """
    with tf.variable_scope(name):
        next_input = inputs
        for i in range(n_layers):
            with tf.variable_scope('conv' + str(i)) as scope:
                normed = layers.layer_norm(next_input)
                nonlinear = leaky_relu(
                    normed, leaky_relu_alpha=leaky_relu_alpha)
                conv = tf.layers.conv1d(
                    inputs=nonlinear,
                    filters=n_filters,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=None,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer()
                )
                next_input = conv
        return next_input * residual_alpha + inputs


def get_var_list(prefix):
    """ to get both Generator's trainable variables and add trainable variables into histogram summary
    
    Params
    ------
    prefix : string
        string to select out the trainalbe variables
    """
    trainable_V = tf.trainable_variables()
    theta = []
    for _, v in enumerate(trainable_V):
        if v.name.startswith(prefix):
            theta.append(v)
            tf.summary.histogram(v.name,
                                 v, collections=[prefix + '_histogram'])
    return theta
