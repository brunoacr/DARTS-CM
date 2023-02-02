import math
import os
from collections import namedtuple
import dill
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *
import tensorflow as tf
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

    @tf.function(jit_compile=True)
    def call(self, inputs, training, *args, **kwargs):
        # TODO [INFO] get_variable accesses a global variable storage where it looks for the variable with these params,
        # TODO [INFO] otherwise it creates a new one. This way, this method can be called multiple times and it will act
        # TODO [INFO] as a persistent object

        weight = tf.nn.softmax(self.arch_weight)
        weight = tf.reshape(weight, [-1, 1])
        outs = []

        for index, op in enumerate(self.ops):
            if PRIMITIVES[index].__contains__('norm'):
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
        self.arch_weights = arch_weights
        self.mixed_ops = None
        self.cells_num = cells_num
        self.multiplier = multiplier
        self.C_out = C_out
        self.prep = \
            None

    def build(self, input_shape):
        self.mixed_ops = []
        for i in range(self.cells_num):
            self.mixed_ops.append(MixedOp(self.C_out, self.arch_weights[i]))

        self.prep = tf.keras.layers.Dense(self.C_out)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, *args, **kwargs):

        x = self.prep(inputs, training=training)

        for i in range(self.cells_num):
            x = self.mixed_ops[i](x, training=training)

        return x


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
        self.prep = None
        self.model_weights = None
        self.step = 0

    def build(self, input_shape):

        # number of mixed_ops in cell 2+3+4+5 ... cells_num times
        self.arch_weights = [None] * self.cells_num
        for i in range(self.cells_num):
            self.arch_weights[i] = (self.add_weight("arch_weight_{}".format(i), [len(PRIMITIVES)],
                                                    initializer=tf.zeros_initializer(),
                                                    trainable=True))

        c_curr = int(math.pow(2, self.layer_num-1) * self.first_C)

        self.cells = []
        for i in range(self.layer_num):
            cell = Cell(self.cells_num, self.multiplier, c_curr, self.arch_weights)
            self.cells.append(cell)
            c_curr = c_curr // 2

        self.prep = tf.keras.layers.Dense(self.first_C)
        self.output_layer = tf.keras.layers.Dense(self.class_num)

    def update(self, clone):
        for v_, v in zip(clone.trainable_weights, self.trainable_weights):
            v_.assign(v)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False, mask=None):
        x = self.prep(inputs)
        for i in range(self.layer_num):
            x = self.cells[i](x)
        return self.output_layer(x)

    def get_clone(self, init):
        if not self.original:
            raise Exception('Non-original main_models cant have clones')
        if self.clone is None:
            self.clone = self.deep_clone(init)
        return self.clone

    @tf.function(jit_compile=True)
    def train_step(self, data):
        batch, labels = data
        x = batch['train']
        x_v = batch['valid']
        y = labels['train']
        y_v = labels['valid']

        # GET TRAIN LOSS + GRADS
        with tf.GradientTape() as tape:
            logits = self(x)
            train_loss = self.compiled_loss(y, logits)

        train_grads = tape.gradient(train_loss, self.get_model_weights())

        # ARCH STEP

        clone = self.get_clone(x)
        self.update(clone)

        # w' =  w − ξ ∇w( L_train(w, α) )
        tf.keras.optimizers.SGD(self.lr(self.step)).apply_gradients(zip(train_grads, clone.get_model_weights()))

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
        opt_neg = tf.keras.optimizers.SGD(-2 * E)

        self.update(clone)
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
            arch_grads[ix] = arch_grads[ix] - self.lr(self.step) * tf.divide(
                train_grads_pos[ix] - train_grads_neg[ix], 2 * E)

        self.arch_opt.apply_gradients(zip(arch_grads, self.arch_weights))  # <-- updating main main_models arch weights

        # MODEL STEP

        with tf.GradientTape() as tape:
            logits = self(x)
            train_loss = self.compiled_loss(y, logits)
        train_grads = tape.gradient(train_loss, self.get_model_weights())

        self.compiled_metrics.update_state(y, logits)
        self.optimizer.apply_gradients(
            zip(train_grads, self.get_model_weights()))  # <-- updating main main_models model weights

        self.step += 1
        return {m.name: m.result() for m in self.metrics}

    def get_arch_weights(self):
        return self.arch_weights

    def get_model_weights(self):
        if self.model_weights is None:
            model_weights = [w for w in self.trainable_weights if (not w.name.__contains__('arch'))]
            self.model_weights = model_weights
        return self.model_weights

    def get_genotype(self):
        def _parse():
            offset = 0
            genotype = []
            arch_var = self.arch_weights

            for i in range(self.cells_num):
                weight = arch_var[i]
                value = weight.numpy()
                value_sorted = value.argsort()
                max_index = value_sorted[-1]

                genotype.extend([(PRIMITIVES[max_index], i)])
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

    def save_model(self, path):
        state = ModelState(original=self.original,
                           lr=self.lr,
                           first_C=self.first_C,
                           multiplier=self.multiplier,
                           cells_num=self.cells_num,
                           layer_num=self.layer_num,
                           class_num=self.class_num,
                           arch_opt=self.arch_opt,
                           arch_weights=self.arch_weights,
                           optimizer=self.optimizer,
                           loss=self.loss
                           )
        dill.dump(state, open(os.path.join(path, 'model_state.pk'), 'wb'))
        self.save_weights(os.path.join(path, 'weights', 'weights.tf'))

    @staticmethod
    def load_model(path):
        state = dill.load(open(os.path.join(path, 'model_state.pk'), 'rb'))
        model = ContinuousModel(arch_opt=state.arch_opt,
                                lr=state.lr,
                                first_C=state.first_C,
                                class_num=state.class_num,
                                layer_num=state.layer_num,
                                cells_num=state.cells_num,
                                multiplier=state.multiplier,
                                original=state.original
                                )

        model.compile(optimizer=state.optimizer, loss=state.loss, metrics=['accuracy'])
        model.load_weights(os.path.join(path, 'weights', 'weights.tf')).expect_partial()
        model.arch_weights = state.arch_weights
        return model


ModelState = namedtuple('ModelState',
                        'original lr first_C multiplier cells_num layer_num class_num arch_opt arch_weights optimizer loss')
