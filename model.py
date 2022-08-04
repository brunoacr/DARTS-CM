import os
import sys

import dill
import numpy as np
import tensorflow as tf
from collections import namedtuple

from operations import *


class Cell(tf.keras.layers.Layer):
    def __init__(self, genotype, cells_num, C_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cells_num = cells_num
        self.ops = None
        self.genotype = genotype
        self.prep0 = None
        self.prep1 = None
        self.C_out = C_out
        self.multiplier = len(genotype.concat)

    def build(self, input_shape):
        op_names, _ = zip(*self.genotype.cell)

        self.ops = []
        for name in op_names:
            self.ops.append(OPS[name](self.C_out))

        self.prep0 = DenseBN(self.C_out, activation='relu')
        self.prep1 = DenseBN(self.C_out, activation='relu')

    def call(self, inputs, training=False, *args, **kwargs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise Exception('Input to Cell Layer must be a list with two items')

        s0, s1 = inputs[0], inputs[1]
        s0 = self.prep0(s0, training=training)
        s1 = self.prep1(s1, training=training)

        _, indices = zip(*self.genotype.cell)
        state = [s0, s1]  # state = [CELL(K-2), CELL(K-1), Node 0, Node 1, Node 2, Node 3 ]
        ix = 0
        for i in range(self.cells_num):
            temp = []
            for j in range(2):
                h = state[indices[2 * i + j]]
                temp.append(self.ops[ix](h, training=training))
                ix += 1

            state.append(tf.add_n(temp))
        out = tf.concat(state[-self.multiplier:], axis=-1)  # Output of cell is concatenation of Nodes 0-3
        return out


class Model(tf.keras.Model):

    def __init__(self, first_C, class_num, layer_num, genotype, cells_num=4, *args, **kwargs):
        super().__init__()
        self.genotype = genotype
        self.first_C = first_C
        self.cells_num = cells_num
        self.layer_num = layer_num
        self.class_num = class_num
        self.output_layer = None
        self.cells = None
        self.s1_pre_proc = None
        self.prep1 = None
        self.prep0 = None
        self.model_weights = None

    def build(self, input_shape):

        # number of mixed_ops in cell 2+3+4+5 ... cells_num times
        C_curr = self.first_C

        self.cells = []
        for i in range(self.layer_num):
            if C_curr > 16:
                C_curr = C_curr // 2
            cell = Cell(self.genotype, self.cells_num, C_curr)
            self.cells.append(cell)

        self.prep0 = DenseBN(self.first_C)
        self.prep1 = DenseBN(self.first_C)
        self.output_layer = tf.keras.layers.Dense(self.class_num)

    def call(self, inputs, training=False, mask=None):
        s0 = self.prep0(inputs)
        s1 = self.prep1(inputs)
        for i in range(self.layer_num):
            s0, s1 = s1, self.cells[i]([s0, s1])
        return self.output_layer(s1)

    def save_model(self, path):
        state = ModelState(genotype=self.genotype,
                           first_C=self.first_C,
                           cells_num=self.cells_num,
                           layer_num=self.layer_num,
                           class_num=self.class_num,
                           optimizer=self.optimizer,
                           loss=self.loss
                           )
        dill.dump(state, open(os.path.join(path, 'model_state.pk'), 'wb'))
        self.save_weights(os.path.join(path, 'weights', 'weights.tf'))

    @staticmethod
    def load_model(path):
        state = dill.load(open(os.path.join(path, 'model_state.pk'), 'rb'))
        model = Model(state.first_C, state.class_num, state.layer_num, state.genotype, state.cells_num)
        model.compile(optimizer=state.optimizer, loss=state.loss, metrics=['accuracy'])
        model.load_weights(os.path.join(path, 'weights', 'weights.tf')).expect_partial()
        return model


ModelState = namedtuple('ModelState', 'first_C class_num layer_num genotype cells_num optimizer loss')
