from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell concat')

PRIMITIVES = [
    'skip_connect',
    'relu',
    'leaky_relu',
    'dropout',
    'gaussian_dropout',
    'batch_norm',
    'layer_norm',
    'dense',
    'dense_relu',
    'dense_leaky_relu'
]
