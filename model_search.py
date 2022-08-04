import os
import sys

import numpy as np
import tensorflow as tf

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


class ContinuousModel(tf.keras.Model):

    def __init__(self, arch_opt, lr, first_C, class_num, layer_num,
                 cells_num=4,
                 multiplier=4,
                 original=True,
                 *args,
                 **kwargs):
        super().__init__()
        self.original = original
        self.clone = None
        self.lr = lr
        self.first_C = first_C
        self.multiplier = multiplier
        self.cells_num = cells_num
        self.layer_num = layer_num
        self.class_num = class_num
        self.arch_opt = arch_opt
        self.output_layer = None
        self.arch_weights = None
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
        self.output_layer = tf.keras.layers.Dense(self.class_num)

    def update(self, clone):
        for v_, v in zip(clone.trainable_weights, self.trainable_weights):
            v_.assign(v)

    def call(self, inputs, training=False, mask=None):
        s0 = self.prep0(inputs)
        s1 = self.prep1(inputs)
        for i in range(self.layer_num):
            s0, s1 = s1, self.cells[i]([s0, s1])
        return self.output_layer(s1)

    def get_clone(self, init):
        if not self.original:
            raise Exception('Non-original models cant have clones')
        if self.clone is None:
            self.clone = self.deep_clone(init)
        return self.clone

    def train_step(self, data):
        batch, labels = data
        cut = int(batch.shape[0] * 0.5)
        x = batch[:cut]
        x_v = batch[cut:]
        y = labels[:cut]
        y_v = labels[cut:]

        # GET TRAIN LOSS + GRADS
        with tf.GradientTape() as tape:
            logits = self(x)
            train_loss = self.compiled_loss(y, logits)

        train_grads = tape.gradient(train_loss, self.get_model_weights())

        # ARCH STEP

        clone = self.get_clone(x)
        self.update(clone)

        # w' =  w − ξ ∇w( L_train(w, α) )
        self.optimizer.apply_gradients(zip(train_grads, clone.get_model_weights()))

        with tf.GradientTape() as tape:
            val_logits = clone(x_v)
            val_loss = self.compiled_loss(y_v, val_logits)

        valid_grads = tape.gradient(val_loss, clone.get_model_weights())  # ∇w'( L_val(w', α) )

        with tf.GradientTape() as tape:
            clone_val_logits = clone(x_v)
            clone_val_loss = self.compiled_loss(y_v, clone_val_logits)

        arch_grads = tape.gradient(clone_val_loss, clone.get_arch_weights())  # ∇α( L_val(w', α))

        e = 1e-2
        E = e / tf.linalg.global_norm(valid_grads)
        opt_pos = tf.keras.optimizers.SGD(E)
        opt_neg = tf.keras.optimizers.SGD(-2*E)

        opt_pos.apply_gradients(zip(valid_grads, clone.get_model_weights()))  # W+ stored in clone
        with tf.GradientTape() as tape:
            pos_train_logits = clone(x)
            pos_train_loss = self.compiled_loss(y, pos_train_logits)

        train_grads_pos = tape.gradient(pos_train_loss, clone.get_arch_weights())  # ∇α L_train(w+, α)

        opt_neg.apply_gradients(zip(valid_grads, clone.get_model_weights()))  # W- stored in clone
        with tf.GradientTape() as tape:
            neg_train_logits = clone(x)
            neg_train_loss = self.compiled_loss(y, neg_train_logits)

        train_grads_neg = tape.gradient(neg_train_loss, clone.get_arch_weights())  # ∇α L_train(w-, α)

        for ix, grad in enumerate(arch_grads):  # ~~ ∇α( L_val(w*, α) ), w* being fully trained model weights
            arch_grads[ix] = arch_grads[ix] - self.lr * tf.divide(
                train_grads_pos[ix] - train_grads_neg[ix], 2 * E)

        self.arch_opt.apply_gradients(zip(arch_grads, self.arch_weights))  # <-- updating main models arch weights

        # MODEL STEP
        with tf.GradientTape() as tape:
            logits = self(x)
            train_loss = self.compiled_loss(y, logits)

        train_grads = tape.gradient(train_loss, self.get_model_weights())
        self.compiled_metrics.update_state(y, logits)
        self.optimizer.apply_gradients(zip(train_grads, self.get_model_weights()))  # <-- updating main models model weights

        return {m.name: m.result() for m in self.metrics}

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
        gene = _parse()
        genotype = Genotype(cell=gene, concat=concat)
        return genotype

    def deep_clone(self, init):
        clone = ContinuousModel(self.arch_opt, self.lr, self.first_C, self.class_num, self.layer_num, self.cells_num,
                                self.multiplier, original=False)
        clone.compile(self.optimizer, self.loss, self.metrics_names)
        clone(init)

        self.update(clone)
        return clone


