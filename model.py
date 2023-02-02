from pathlib import Path

import dill
from collections import namedtuple
import visualize
from operations import *


class Cell(tf.keras.layers.Layer):
    def __init__(self, genotype, cells_num, C_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cells_num = cells_num
        self.ops = None
        self.genotype = genotype
        self.prep = None
        self.C_out = C_out
        self.multiplier = len(genotype.concat)

    def build(self, input_shape):
        op_names, _ = zip(*self.genotype.cell)

        self.ops = []
        for name in op_names:
            self.ops.append(OPS[name](self.C_out))

        self.prep = tf.keras.layers.Dense(self.C_out)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, *args, **kwargs):

        x = self.prep(inputs)

        for i in range(self.cells_num):
            x = self.ops[i](x, training=training)

        return x


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
        self.prep = None
        self.model_weights = None

    def build(self, input_shape):

        # number of mixed_ops in cell 2+3+4+5 ... cells_num times
        # c_curr = int(math.pow(2, self.layer_num-1) * self.first_C)
        c_curr = self.first_C
        self.cells = []
        for i in range(self.layer_num):
            cell = Cell(self.genotype, self.cells_num, c_curr)
            self.cells.append(cell)
            # c_curr = c_curr // 2
        self.prep = tf.keras.layers.Dense(self.first_C)
        self.output_layer = tf.keras.layers.Dense(self.class_num)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, mask=None):
        # x = self.prep(inputs)
        x = inputs
        for i in range(self.layer_num):
            x = self.cells[i](x)
        return self.output_layer(x)

    def save_model(self, path):
        state = ModelState(genotype=self.genotype,
                           first_C=self.first_C,
                           cells_num=self.cells_num,
                           layer_num=self.layer_num,
                           class_num=self.class_num,
                           optimizer=self.optimizer,
                           loss=self.loss
                           )
        dill.dump(state, open(Path(path, 'model_state.pk'), 'wb'))
        self.save_weights(Path(path, 'weights', 'weights.tf'))
        with open(Path(path, 'genotype.txt'), 'w') as file:
            file.write(str(self.genotype))
        dill.dump(self.genotype, open(Path(path, 'genotype.pk'), 'wb'))
        visualize.plot(self.genotype.cell, Path(path, 'genotype'))

    @staticmethod
    def load_model(path):
        state = dill.load(open(Path(path, 'model_state.pk'), 'rb'))
        model = Model(state.first_C, state.class_num, state.layer_num, state.genotype, state.cells_num)
        model.compile(optimizer=state.optimizer, loss=state.loss, metrics=['accuracy'])
        model.load_weights(Path(path, 'weights', 'weights.tf')).expect_partial()
        return model


ModelState = namedtuple('ModelState', 'first_C class_num layer_num genotype cells_num optimizer loss')
