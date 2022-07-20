import os
import sys

import numpy as np
import tensorflow as tf
from mapping_network_utils import ActivationsExtractor

from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *
import utils

null_scope = tf.compat.v1.VariableScope("")


class MixedOp(tf.keras.layers.Layer):
    def __init__(self, C_out, arch_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops = None
        self.C_out = C_out
        self.arch_weight = arch_weight

    def build(self, input_shape):
        self.ops = []
        for primitive in PRIMITIVES:
            self.ops.append(OPS[primitive](self.C_out))

    def call(self, inputs, training, *args, **kwargs):
        # TODO [INFO] get_variable accesses a global variable storage where it looks for the variable with these params,
        # TODO [INFO] otherwise it creates a new one. This way, this method can be called multiple times and it will act
        # TODO [INFO] as a persistent object

        weight = nn.softmax(self.arch_weight)
        weight = tf.reshape(weight, [-1, 1])

        outs = []

        for index, op in enumerate(self.ops):
            if PRIMITIVES[index] == 'batch_norm':
                out = op(inputs, training=training)
            else:
                out = op(inputs)
            mask = [i == index for i in range(len(PRIMITIVES))]  # TODO [INFO] getting the arch weight for current op
            w_mask = tf.constant(mask, tf.bool)
            w = tf.boolean_mask(tensor=weight, mask=w_mask)
            outs.append(out * w)  # TODO [INFO] getting the op feature map weighed by the arch weights

        return tf.add_n(outs)  # TODO [INFO] returning the sum of all partial feature maps


class Cell(tf.keras.layers.Layer):
    def __init__(self, cells_num, multiplier, C_out, arch_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prep0 = None
        self.prep1 = None
        self.arch_weights = arch_weights
        self.mixed_ops = None
        self.cells_num = cells_num
        self.multiplier = multiplier
        self.C_out = C_out

    def build(self, input_shape):
        self.mixed_ops = []
        for i in range(tf.add_n(range(2, self.cells_num + 2)).numpy()):
            self.mixed_ops.append(MixedOp(self.C_out, self.arch_weights[i]))

        self.prep0 = DenseBN(self.C_out, activation='relu')
        self.prep1 = DenseBN(self.C_out, activation='relu')

    def call(self, inputs, training=False, *args, **kwargs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise Exception('Input to Cell Layer must be a list with two items')

        s0, s1 = inputs[0], inputs[1]
        s0 = self.prep0(s0, training=training)
        s1 = self.prep1(s1, training=training)

        state = [s0, s1]  # state = [CELL(K-2), CELL(K-1), Node 0, Node 1, Node 2, Node 3 ]
        ix = 0
        for i in range(self.cells_num):
            temp = []
            for j in range(2 + i):
                temp.append(self.mixed_ops[ix](state[j], training=training))
                ix += 1

            state.append(tf.add_n(temp))
        out = tf.concat(state[-self.multiplier:], axis=-1)  # Output of cell is concatenation of Nodes 0-3
        return out


class Model(tf.keras.Model):

    def __init__(self, loss_fn, first_C, class_num, layer_num, cells_num=4, multiplier=4, *args,
                 **kwargs):
        super().__init__()
        self.loss_fn = loss_fn
        self.first_C = first_C
        self.multiplier = multiplier
        self.cells_num = cells_num
        self.layer_num = layer_num
        self.class_num = class_num
        self.dense = None
        self.arch_weights = None
        self.s0_pre_proc = None
        self.cells = None
        self.s1_pre_proc = None
        self.prep1 = None
        self.prep0 = None
        self.model_weights = None

    def build(self, input_shape):
        # number of mixed_ops in cell 2+3+4+5 ... cells_num times
        self.arch_weights = [None] * tf.add_n(range(2, self.cells_num + 2)).numpy()
        for i, _ in enumerate(self.arch_weights):
            self.arch_weights[i] = self.add_weight("arch_weight_{}".format(i), [len(PRIMITIVES)],
                                                   initializer=tf.compat.v1.random_normal_initializer(0, 1e-3),
                                                   trainable=True)
        C_curr = self.first_C

        self.cells = []
        for i in range(self.layer_num):
            if C_curr > 16:
                C_curr = C_curr // 2
            cell = Cell(self.cells_num, self.multiplier, C_curr, self.arch_weights)
            self.cells.append(cell)

        self.prep0 = DenseBN(self.first_C)
        self.prep1 = DenseBN(self.first_C)
        self.dense = tf.keras.layers.Dense(self.class_num, input_shape = input_shape, activation='sigmoid' if self.class_num == 1 else 'softmax')

    def call(self, inputs, training=False, mask=None):
        x = inputs
        s0 = self.prep0(x)
        s1 = self.prep1(x)
        for i in range(self.layer_num):
            s0, s1 = s1, self.cells[i]([s0, s1])
        logits = self.dense(inputs)
        return logits

    def get_arch_weights(self):
        return self.arch_weights

    def get_model_weights(self):
        if self.model_weights is None:
            model_weights = [w for w in self.trainable_weights if (not w.name.__contains__('arch'))]
            self.model_weights = model_weights
        return self.model_weights

    def get_genotype(self):
        # todo [info] -> for each node in the search space, returns the two incoming mixed-ops which have the most confidence in their choice of op
        def _parse():
            offset = 0
            genotype = []
            arch_var = self.arch_weights
            for i in range(self.cells_num):
                edges = []
                edges_confident = []
                for j in range(i + 2):
                    weight = arch_var[offset + j]
                    value = weight.numpy()
                    value_sorted = value.argsort()
                    max_index = value_sorted[-2] if value_sorted[-1] == PRIMITIVES.index('none') else value_sorted[-1]

                    edges.append((PRIMITIVES[max_index], j))
                    edges_confident.append(value[max_index])
                edges_confident = np.array(edges_confident)
                max_edges = [edges[np.argsort(edges_confident)[-1]], edges[np.argsort(edges_confident)[-2]]]
                genotype.extend(max_edges)
                offset += i + 2
            return genotype

        concat = list(range(2 + self.cells_num - self.multiplier, self.cells_num + 2))
        gene_normal = _parse()
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype

    def deep_clone(self, init):
        clone = Model(self.loss_fn, self.first_C, self.class_num, self.layer_num, self.cells_num, self.multiplier)
        clone(init, training=True)  # builds model
        for v_, v in zip(clone.get_model_weights(), self.get_model_weights()):
            v_.assign(v)
        return clone

    def compute_loss(self, x, y, weight_decay=1e-3, return_logits=False):
        logits = self(x, training=True)
        loss = self.loss_fn(y, logits)
        reg = l2(0.0001)
        reg_loss = [reg(v) for v in self.trainable_weights]
        reg_loss = tf.reduce_mean(reg_loss)
        loss += 1e4 * weight_decay * reg_loss
        if return_logits:
            return logits, loss
        else:
            return loss


def Model_test(x, y, is_training, name="weight_var"):
    weight_decay = 3e-4
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], activation_fn=None, padding='SAME',
                            biases_initializer=None, weights_regularizer=tf.keras.regularizers.l2(0.5 * (0.0001))):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                # x=slim.max_pool2d(x,[3,3],stride=2)
                s0 = x
                s1 = x
                for i in range(1):
                    s0, s1 = s1, Cell(s0, s1, 4, 4, 32, False, False)
                out = tf.reduce_mean(input_tensor=s1, axis=[1, 2], keepdims=True, name='global_pool')
                logits = slim.conv2d(out, 10, [1, 1], activation_fn=None, normalizer_fn=None,
                                     weights_regularizer=tf.keras.regularizers.l2(0.5 * (0.0001)))
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
    train_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    return logits, train_loss
