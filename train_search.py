import math
import os
import tensorflow as tf
import sys
import time
import glob
import numpy as np
import logging
import argparse

import visualize
from model_search import *
from data_utils import read_data
from datetime import datetime
import utils
from mapping_network_utils.activations_extractor import ActivationsExtractor
from main_network.main_network_utils import load_dataset
import more_itertools

# PARAMS
EXTRACTOR_PATH = './extractors/xtrains_extractor'
DATA_PATH = './data/XTRAINS_MINI'
DATASET = 'XTRAINS'
SEC_LABELS = ['WarTrain']


# De-comment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def parse_args():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="location of the data corpus")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
    parser.add_argument("--learning_rate_min", type=float, default=0.25, help="min learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
    parser.add_argument("--report_freq", type=float, default=200, help="report frequency")
    parser.add_argument("--gpu", type=int, default=1, help="gpu device id")
    parser.add_argument("--epochs", type=int, default=100, help="num of training epochs")
    parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
    parser.add_argument("--layers", type=int, default=3, help="total number of layers")
    parser.add_argument("--model_path", type=str, default="saved_models", help="path to save the model")
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    parser.add_argument("--drop_path_prob", type=float, default=0.3, help="drop path probability")
    parser.add_argument("--save", type=str, default="EXP", help="experiment name")
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--train_portion", type=float, default=0.8, help="portion of training data")
    parser.add_argument("--unrolled", action="store_true", default=False,
                        help="use one-step unrolled validation loss", )
    parser.add_argument("--arch_learning_rate", type=float, default=3e-4, help="learning rate for arch encoding")
    parser.add_argument("--arch_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding")

    # BR args - start
    parser.add_argument('--extractor', type=str, default=EXTRACTOR_PATH,
                        help='location of the activations extractor')
    parser.add_argument('--sec_labels', type=str, default=SEC_LABELS, nargs='+',
                        help='Secondary labels mapping model should learn')
    parser.add_argument('--dataset', type=str, default=DATASET, choices=['VCAB', 'XTRAINS'])
    # BR args - end
    return parser.parse_args()


args = parse_args()
tf.compat.v1.set_random_seed(args.seed)
output_dir = "./outputs/train_model/"
if not os.path.isdir(output_dir):
    print("Path {} does not exist. Creating.".format(output_dir))
    os.makedirs(output_dir)

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
CLASS_NUM = len(args.sec_labels)
print('CLASS_NUM: {}'.format(CLASS_NUM))


def pad(a, div):
    if len(a) % div == 0:
        return a
    ix = range(len(a))
    choices = np.random.choice(ix, size=div - (len(a) % div))
    a.extend([a[c] for c in choices])
    return a


def preprocess(x, y, batch_size):
    x, y = shuffle_together(x, y)
    x = batch(x, batch_size)
    y = batch(y, batch_size)
    return tf.constant(np.stack(x), dtype='float32'), np.stack(y)


def batch(a, batch_size):
    a = pad(a, batch_size)
    out = [a[0:batch_size]]
    for i in range(batch_size, len(a), batch_size):
        ap = a[i: i + batch_size]
        out.append(ap)
    return out


def shuffle_together(a, b):
    assert len(a) == len(b)
    p = list(np.random.permutation(len(a)))
    a = [a[i] for i in p]
    b = [b[i] for i in p]
    return a, b


def main():
    images, labels = load_dataset(args.dataset, args.data, val_split=1 - args.train_portion, test_split=0,
                                  sec_labels=True,
                                  filter_sec_labels=args.sec_labels)

    x_train, y_train = preprocess(images['train'], labels['train'], args.batch_size)
    x_val, y_val = preprocess(images['valid'], labels['valid'], args.batch_size)

    model_loss = tf.metrics.binary_crossentropy
    main_model = Model(model_loss, args.init_channels, CLASS_NUM, args.layers)

    extractor = ActivationsExtractor.load(args.extractor)  # load extractor

    init = extractor.extract_activations(x_train[0])
    main_model(init)  # builds model
    main_model.summary()
    print(CLASS_NUM)
    lr = tf.keras.optimizers.schedules.CosineDecay(
        args.learning_rate,
        ((len(x_train) + len(x_val)) / args.batch_size) * args.epochs,
        args.learning_rate_min
    )

    arch_opt = tf.keras.optimizers.Adam(args.arch_learning_rate, 0.5, 0.999)
    model_opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=args.momentum)

    if not tf.test.is_gpu_available():
        raise Warning('Tensorflow not using GPU!')

    genotype_record_file = open(output_dir + "genotype_record_file.txt", "w")
    gs = 0
    for epoch in range(args.epochs):
        print('start of an epoch')
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        test_top1 = utils.AvgrageMeter()

        for i in range(len(x_train)):
            x, y = extractor.extract_activations(x_train[i]), y_train[i]
            x_v, y_v = extractor.extract_activations(x_val[i % len(x_val)]), y_val[i % len(y_val)]

            with tf.GradientTape(persistent=True) as tape:
                # get train loss
                # logits, train_loss = main_model.compute_loss(x, y, args.weight_decay, return_logits=True)
                logits = main_model.call(x)
                print('logits = {}'.format(logits))
                print('label  = {}'.format(y))
                train_loss = model_loss(y,logits)
                print('train loss = {}'.format(train_loss))
                with tape.stop_recording():
                    watched = tape.watched_variables()
                    train_grads = tape.gradient(train_loss, main_model.get_model_weights())

                    # ARCH STEP
                    clone = main_model.deep_clone(init)
                    model_opt.apply_gradients(zip(train_grads, clone.get_model_weights()))  # w' =  w − ξ ∇w( L_train(w, α) )

                    valid_loss = clone.compute_loss(x_v, y_v, args.weight_decay)

            valid_grads = tape.gradient(valid_loss, clone.get_model_weights())  # ∇w'( L_val(w', α) )

            e = 1e-2
            E = e / tf.linalg.global_norm(valid_grads)
            clone_pos = clone.deep_clone(init)
            clone_neg = clone.deep_clone(init)
            opt_pos = tf.keras.optimizers.SGD(E)
            opt_pos.apply_gradients(zip(valid_grads, clone_pos.get_model_weights()))  # W+ stored in clone_pos
            opt_neg = tf.keras.optimizers.SGD(-E)
            opt_neg.apply_gradients(zip(valid_grads, clone_neg.get_model_weights()))  # W- stored in clone_neg

            with tf.GradientTape(persistent=True) as tape:
                pos_train_loss = clone_pos.compute_loss(x, y, args.weight_decay)
                neg_train_loss = clone_neg.compute_loss(x, y, args.weight_decay)

            train_grads_pos = tape.gradient(pos_train_loss, clone_pos.get_arch_weights())  # ∇α L_train(w+, α)
            train_grads_neg = tape.gradient(neg_train_loss, clone_neg.get_arch_weights())  # ∇α L_train(w-, α)

            with tf.GradientTape() as tape:
                clone_val_loss = clone.compute_loss(x_v, y_v, args.weight_decay)

            arch_grads = tape.gradient(clone_val_loss, clone.get_arch_weights())  # ∇α( L_val(w', α))

            for ix, grad in enumerate(arch_grads):  # ~~ ∇α( L_val(w*, α) ), w* being fully trained model weights
                arch_grads[ix] = arch_grads[ix] - args.learning_rate * tf.divide(
                    train_grads_pos[ix] - train_grads_neg[ix], 2 * E)

            arch_opt.apply_gradients(
                zip(arch_grads, main_model.get_arch_weights()))  # <-- updating main models arch weights

            # MODEL STEP
            model_opt.apply_gradients(zip(train_grads, main_model.get_model_weights()))
            acc = tf.reduce_mean(
                input_tensor=tf.cast(tf.nn.in_top_k(predictions=logits, targets=np.asarray(y).flatten(), k=1),
                                     tf.float32))

            objs.update(np.mean(train_loss), args.batch_size)
            top1.update(acc, args.batch_size)

            if gs % args.report_freq == 0 or True:
                print(
                    "epochs {} steps {} currnt lr is {:.3f}  loss is {}  train_acc is {}".format(
                        epoch, gs, lr(epoch), objs.avg, top1.avg
                    )
                )
            gs += 1

        print("-" * 80)
        print("end of an epoch")

        genotype = main_model.get_genotype()
        print("genotype is {}".format(genotype))
        genotype_record_file.write("{}".format(genotype) + "\n")

        for i in range(len(x_val)):
            x_v, y_v = extractor.extract_activations(x_val[i]), y_val[i]
            logits = main_model(x_v)
            valid_acc = tf.reduce_mean(
                input_tensor=tf.cast(tf.nn.in_top_k(predictions=logits, targets=np.asarray(y_v).flatten(), k=1),
                                     tf.float32))
            test_top1.update(valid_acc, args.batch_size)

        print(
            "******************* epochs {}   valid_acc is {}".format(
                epoch, test_top1.avg
            )
        )
        print("-" * 80)
        print("end of a valid epoch")


if __name__ == "__main__":
    main()
