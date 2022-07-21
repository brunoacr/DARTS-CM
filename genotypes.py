from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'batch_norm',
    'dropout',
    'dense'
    # 'max_pool_3x3',
    # 'avg_pool_3x3',
    # 'sep_conv_3x3',
    # 'sep_conv_5x5',
    # 'dil_conv_3x3',
    # 'dil_conv_5x5'
]
