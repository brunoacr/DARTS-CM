import os
import sys
from tensorflow.python.ops import nn
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

OPS = {
    'none': lambda C: Zero(C),
    'skip_connect': lambda C: Identity(),
    'batch_norm': lambda C: tf.keras.layers.BatchNormalization(),
    'dropout': lambda C: tf.keras.layers.Dropout(rate=0.1),
    'dense_relu': lambda C: tf.keras.layers.Dense(C, activation='relu'),
    'dense': lambda C: tf.keras.layers.Dense(C)
}


class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x):
        return x


class Zero(tf.keras.layers.Layer):
    def __init__(self, C_out):
        super(Zero, self).__init__()
        self.C_out = C_out

    def call(self, x):
        return tf.zeros_like(x)[:, :self.C_out]


class DenseBN(tf.keras.layers.Layer):
    def __init__(self, C_out, activation=None):
        super(DenseBN, self).__init__()
        self.activation = activation
        self.C_out = C_out
        self.BN = None
        self.dense = None

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.C_out, self.activation, name='DenseBN')
        self.BN = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.dense(x)
        x = self.BN(x, training=training)
        return x

# def DilConv(x, C_out, kernel_size, stride, rate):
#     x = nn.relu(x)
#     x = slim.separable_convolution2d(x, C_out, kernel_size, depth_multiplier=1, stride=stride, rate=rate)
#     x = slim.batch_norm(x)
#     return x
#
#
# def SepConv(x, C_out, kernel_size, stride):
#     x = nn.relu(x)
#     C_in = x.get_shape()[-1]
#
#     x = slim.separable_convolution2d(x, C_in, kernel_size, depth_multiplier=1, stride=stride)
#     x = slim.batch_norm(x)
#
#     x = slim.separable_convolution2d(x, C_out, kernel_size, depth_multiplier=1)
#     x = slim.batch_norm(x)
#     return x
#
#
# def FactorizedReduce(x, c_out):
#     x = nn.relu(x)
#     conv1 = slim.conv2d(x, c_out // 2, [1, 1], stride=[2, 2])
#     conv2 = slim.conv2d(x[:, 1:, 1:, :], c_out // 2, [1, 1], stride=[2, 2])
#     x = tf.concat([conv1, conv2], -1)
#     x = slim.batch_norm(x)
#     return x
#
#
# def ReLUConvBN(x, C_out):
#     x = nn.relu(x)
#     x = slim.conv2d(x, C_out, [1, 1])
#     x = slim.batch_norm(x)
#     return x
