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

tf.config.run_functions_eagerly(True)

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
    # x = batch(x, batch_size)
    # y = batch(y, batch_size)
    return np.stack(x), np.stack(y)


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
    if not tf.test.is_gpu_available():
        raise Warning('Tensorflow not using GPU!')

    images, labels = load_dataset(args.dataset, args.data, val_split=1 - args.train_portion, test_split=0,
                                  sec_labels=True,
                                  filter_sec_labels=args.sec_labels)

    extractor = ActivationsExtractor.load(args.extractor)  # load extractor
    extract_vec = np.vectorize(extractor.extract_activations)

    x_train = np.array([extractor.extract_activations(np.expand_dims(i, axis=0)) for i in images['train']])
    x_val = np.array([extractor.extract_activations(np.expand_dims(i, axis=0)) for i in images['valid']])
    y_train = labels['train']
    y_val = labels['valid']


    model_loss = tf.metrics.binary_crossentropy

    lr = tf.keras.optimizers.schedules.CosineDecay(
        args.learning_rate,
        ((len(x_train) + len(x_val)) / args.batch_size) * args.epochs,
        args.learning_rate_min
    )

    arch_opt = tf.keras.optimizers.Adam(args.arch_learning_rate, 0.5, 0.999)
    model_opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=args.momentum)

    main_model = Model(arch_opt,
                       args.learning_rate,
                       args.init_channels,
                       CLASS_NUM,
                       args.layers)

    main_model.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    main_model(np.expand_dims(x_train[0], axis=0))  # builds model
    main_model.summary()

    main_model.fit(x_train, y_train, epochs=args.epochs, batch_size=int((args.batch_size * 100) / 80),
                   validation_data=zip(np.expand_dims(x_val, axis=0), np.expand_dims(y_val, axis=0)))

    genotype_record_file = open(output_dir + "genotype_record_file.txt", "w")
    gs = 0


if __name__ == "__main__":
    main()
