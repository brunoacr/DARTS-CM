import argparse
import math

# external
from pathlib import Path

import numpy as np
from datetime import datetime
import os
import tensorflow as tf

# internal
from utils.activations_extractor import ActivationsExtractor
from utils.data_utils import loadDataset, loadExtractor
import visualize
from utils import callbacks
from model_search import ContinuousModel
from model import Model
import sys

sys.path.append('G:\System Folders/University\Dissertação\Projects\MAPPING_UTILS\mapping_network_utils')


# PARAMS

# De-comment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, default=FOLDER_NAME, help="desired name for output folder")
    parser.add_argument("--complexity", type=int, default=COMPLEXITY, help="desired dataset complexity")
    parser.add_argument("--main_network", type=str, default=MAIN_NETWORK,
                        choices=['Residential', 'Commercial', 'Industrial'], help="main network to extract from")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=LR, help="init learning rate")
    parser.add_argument("--learning_rate_min", type=float, default=MIN_LR, help="min learning rate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="num of training epochs")
    parser.add_argument("--init_neurons", type=int, default=INIT_NEURONS, help="num of init channels")
    parser.add_argument("--layers", type=int, default=LAYERS, help="total number of layers")
    parser.add_argument("--arch_learning_rate", type=float, default=ARCH_LR, help="learning rate for arch encoding")
    parser.add_argument("--arch_weight_decay", type=float, default=ARCH_DECAY, help="weight decay for arch encoding")
    parser.add_argument("--val_portion", type=float, default=VAL_PORTION, help="portion of training data")
    parser.add_argument("--test_portion", type=float, default=TEST_PORTION, help="portion of training data")
    parser.add_argument("--dataset_size", type=int, default=DATASET_SIZE, help="portion of training data")
    parser.add_argument('--concept', type=str, default=CONCEPT, help='Concept to Map')
    return parser.parse_args()


def convert_to_activations(images):
    for key in images:
        images[key] = np.array([args.extractor.extract_activations(np.expand_dims(i, axis=0)) for i in images[key]])


def architecture_search():
    x_train, y_train = args.images['train'], args.labels['train']
    x_val, y_val = args.images['valid'], args.labels['valid']

    main_model = ContinuousModel(args.arch_opt,
                                 args.lr,
                                 args.init_neurons,
                                 1,
                                 args.layers)

    main_model.compile(loss=args.loss_fn,
                       optimizer=args.model_opt,
                       metrics=['accuracy'])

    main_model(np.expand_dims(x_train[0], axis=0))  # builds model

    c_dir = Path(args.output_dir, 'Continous')
    model_dir = Path(c_dir, 'model')
    os.makedirs(model_dir)

    saver = callbacks.SaveBestModel(model_dir)
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss')
    metrics_logger = callbacks.MetricsLogger(c_dir, name='Continuous')
    genotype_logger = callbacks.GenotypeLogger(c_dir, name='Continuous')

    x = {'train': x_train, 'valid': x_val}
    y = {'train': y_train, 'valid': y_val}
    main_model.fit(x, y,
                   epochs=args.epochs,
                   batch_size=args.batch_size * 2,
                   validation_data=(x_val, y_val),
                   callbacks=[es, metrics_logger, genotype_logger, saver])

    main_model = ContinuousModel.load_model(model_dir)
    return main_model.get_genotype()


def train_discrete_model(genotype, layer_num):
    images, labels, model_opt, lr = args.images, args.labels, args.model_opt, args.lr
    x_train, y_train, x_val, y_val, x_test, y_test = images['train'], labels['train'], images['valid'], labels['valid'], \
                                                     images['test'], labels['test']

    model = Model(int(math.pow(2, layer_num) * args.init_neurons),
                  1,
                  layer_num,
                  genotype)

    model.compile(loss=args.loss_fn,
                  optimizer=model_opt,
                  metrics=['accuracy'])

    d_dir = os.path.join(args.output_dir, 'Discrete_{}Layers'.format(layer_num))
    model_dir = os.path.join(d_dir, 'model')
    os.makedirs(model_dir)

    es = tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_loss')
    logger = callbacks.MetricsLogger(d_dir, name='Discrete_' + str(layer_num) + '_')
    saver = callbacks.SaveBestModel(model_dir)

    model.fit(x_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=(x_val, y_val),
              callbacks=[es, logger, saver])

    model = Model.load_model(model_dir)

    metrics = model.evaluate(x_test, y_test, return_dict=True)
    print('Trained Model Test Metrics: {}'.format(str(metrics)))

    file = open(os.path.join(model_dir, 'eval.txt'), 'w')
    file.write(str(metrics))
    file.close()


def prepare():

    #  -------  OUTPUT DIR  -------

    args.output_dir = Path('./outputs', args.folder_name, 'C' + str(args.complexity) + '_' + str(args.main_network),
                           args.concept,
                           TIMESTAMP)

    if not os.path.isdir(args.output_dir):
        print("Creating output dir: {}".format(args.output_dir))
        os.makedirs(args.output_dir, exist_ok=True)

    #  -------   CONFIG FILE  -------
    config_file = open(os.path.join(args.output_dir, 'config.txt'), 'w')
    config_file.write(str(args))
    config_file.close()

    #  -------   LOAD DATA  -------

    args.images, args.labels = loadDataset(args.complexity,
                                           balance_class=args.concept,
                                           size=args.dataset_size,
                                           val_split=args.val_portion,
                                           test_split=args.test_portion,
                                           padding=True
                                           )

    #  -------   EXTRACTOR  -------

    args.extractor = loadExtractor(args.complexity, args.main_network)
    convert_to_activations(args.images)

    #  -------   LEARNING RATE SCHEDULER  -------

    args.lr = tf.keras.optimizers.schedules.CosineDecay(
        args.learning_rate,
        (len(args.images['train']) / args.batch_size) * args.epochs,
        args.learning_rate_min
    )

    #  -------   OPTIMIZERS  -------
    # args.model_opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=args.momentum)
    args.model_opt = tf.keras.optimizers.Adam()

    args.arch_opt = tf.keras.optimizers.Adam(args.arch_learning_rate, 0.5, 0.999)

    #  -------   LOSS FUNCTION  -------

    args.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def main():
    if not tf.test.is_gpu_available():
        args.logger.warning('Tensorflow not using GPU')

    prepare()
    print('Starting Architecture Search')
    genotype = architecture_search()
    visualize.plot(genotype.cell, os.path.join(args.output_dir, 'genotype'))

    print('Architecture Search Finished \n Genotype: {}'.format(
        genotype))
    for layer_num in range(1, 5):
        print('-' * 80, 'Training discrete model with {} layers'.format(layer_num))
        train_discrete_model(genotype, layer_num)

    print('Finished Discreet Model Training \n Logs and trained models saved at {}'.format(args.output_dir))


MAIN_NETWORK = 'Commercial'
FOLDER_NAME = 'Experience-name'
COMPLEXITY = 3
BATCH_SIZE = 32
LR = 0.003
MIN_LR = 0.0008
EPOCHS = 1
INIT_NEURONS = 16
LAYERS = 4
CONCEPT = 'Billboard'
VAL_PORTION = 0.25
TEST_PORTION = 0.5
DATASET_SIZE = 100
OUTPUT_DIR = "./outputs/"
ARCH_LR = 3e-4
ARCH_DECAY = 1e-3

if __name__ == "__main__":
    args = parse_args()
    TIMESTAMP = datetime.now().strftime('%d%m_%H%M%S')
    main()
