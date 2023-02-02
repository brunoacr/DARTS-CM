import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU, Dropout, BatchNormalization, GaussianDropout, \
    LayerNormalization
from tensorflow.keras.models import Sequential


OPS = {
    'skip_connect': lambda C: Identity(),
    'relu': lambda C: ReLU(),
    'leaky_relu': lambda C: LeakyReLU(),
    'dropout': lambda C: Dropout(0.3),
    'gaussian_dropout': lambda C: GaussianDropout(0.3),
    'batch_norm': lambda C: BatchNormalization(),
    'layer_norm': lambda C: LayerNormalization(),
    'dense': lambda C: Dense(C),
    'dense_relu': lambda C: Sequential([Dense(C), ReLU()]),
    'dense_leaky_relu': lambda C: Sequential([Dense(C), LeakyReLU()])
}


class Function(tf.keras.layers.Layer):
    def __init__(self, function):
        super(Function, self).__init__()
        self.function = function

    def call(self, inputs, *args, **kwargs):
        return self.function(inputs)


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
