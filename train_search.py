import argparse

import numpy as np

from model_search import *
from model import *
from datetime import datetime
from mapping_network_utils.activations_extractor import ActivationsExtractor
from main_network.main_network_utils import load_dataset
import tensorflow as tf
import callbacks

# PARAMS

# De-comment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_PATH, help="location of the data corpus")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="init learning rate")
    parser.add_argument("--learning_rate_min", type=float, default=0.001, help="min learning rate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="num of training epochs")
    parser.add_argument("--init_channels", type=int, default=1024, help="num of init channels")
    parser.add_argument("--layers", type=int, default=LAYERS, help="total number of layers")
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
    parser.add_argument("--drop_path_prob", type=float, default=0.3, help="drop path probability")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--arch_learning_rate", type=float, default=3e-4, help="learning rate for arch encoding")
    parser.add_argument("--arch_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding")
    parser.add_argument("--val_portion", type=float, default=VAL_PORTION, help="portion of training data")
    parser.add_argument("--test_portion", type=float, default=TEST_PORTION, help="portion of training data")
    parser.add_argument("--dataset_size", type=float, default=DATASET_SIZE, help="portion of training data")
    parser.add_argument('--extractor_path', type=str, default=EXTRACTOR_PATH,
                        help='location of the activations extractor')
    parser.add_argument('--sec_labels', type=str, nargs='+', default=['WarTrain'],
                        help='Secondary labels mapping model should learn')
    parser.add_argument('--dataset', type=str, default=DATASET, choices=['VCAB', 'XTRAINS'])
    return parser.parse_args()


def convert_to_activations(images):
    for key in images:
        images[key] = np.array([args.extractor.extract_activations(np.expand_dims(i, axis=0)) for i in images[key]])


def architecture_search():
    x_train, y_train = args.images['train'], args.labels['train']
    x_val, y_val = args.images['valid'], args.labels['valid']

    main_model = ContinuousModel(args.arch_opt,
                                 args.lr.initial_learning_rate,
                                 args.init_channels,
                                 CLASS_NUM,
                                 args.layers)

    main_model.compile(loss=args.loss_fn,
                       optimizer=args.model_opt,
                       metrics=['accuracy'])
    main_model(np.expand_dims(x_train[0], axis=0))  # builds model
    main_model.summary()

    es = tf.keras.callbacks.EarlyStopping(patience=25, monitor='val_loss')

    metrics_logger = callbacks.MetricsLogger(args.output_dir, name='Continuous')
    genotype_logger = callbacks.GenotypeLogger(args.output_dir, name='Continuous')

    main_model.fit(x_train, y_train,
                   epochs=args.epochs,
                   batch_size=int((args.batch_size * 100) / 50),
                   validation_data=(x_val, y_val),
                   callbacks=[es, metrics_logger, genotype_logger])

    return main_model.get_genotype()


def train_discreet_model(genotype):
    images, labels, model_opt, lr = args.images, args.labels, args.model_opt, args.lr
    x_train, y_train, x_val, y_val, x_test, y_test = images['train'], labels['train'], images['valid'], labels['valid'], \
                                                     images['test'], labels['test']

    model = Model(args.init_channels,
                  CLASS_NUM,
                  args.layers,
                  genotype)

    model.compile(loss=args.loss_fn,
                  optimizer=model_opt,
                  metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True)
    logger = callbacks.MetricsLogger(args.output_dir, name='Discreet')

    model.fit(x_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=(x_val, y_val),
              callbacks=[es, logger])

    metrics = model.evaluate(x_test, y_test, return_dict=True)
    print('Trained Model Test Metrics: {}'.format(str(metrics)))
    save_model(model, metrics)


def save_model(model, metrics):
    model_dir = os.path.join(args.output_dir, 'trained_model')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    file = open(os.path.join(model_dir, 'eval.txt'), 'w')
    file.write(str(metrics))
    file.close()
    model.save_model(model_dir)


def prepare():
    #  -------  OUTPUT DIR  -------
    args.output_dir = os.path.join('./outputs', args.dataset, args.extractor_path.split('/')[-1], str(args.sec_labels),
                                   str(args.layers) + '_Layers_' + TIMESTAMP)

    if not os.path.isdir(args.output_dir):
        print("Creating output dir: {}".format(args.output_dir))
        os.makedirs(args.output_dir, exist_ok=True)

    #  -------   CONFIG FILE  -------
    config_file = open(os.path.join(args.output_dir, 'config.txt'), 'w')
    config_file.write(str(args))
    config_file.close()

    #  -------   LOAD DATA  -------

    args.images, args.labels = load_dataset(args.dataset, args.data, size=args.dataset_size, val_split=args.val_portion,
                                            test_split=args.test_portion,
                                            sec_labels=True,
                                            filter_sec_labels=args.sec_labels)

    print('Train size: {} Val Size: {} Test Size: {}'.format(len(args.images['train']), len(args.images['valid']),
                                                             len(args.images['test'])))
    #  -------   EXTRACTOR  -------
    args.extractor = ActivationsExtractor.load(args.extractor_path)  # load extractor
    convert_to_activations(args.images)

    #  -------   LEARNING RATE SCHEDULER  -------

    args.lr = tf.keras.optimizers.schedules.CosineDecay(
        args.learning_rate,
        (len(args.images['train']) / args.batch_size) * args.epochs,
        args.learning_rate_min
    )

    #  -------   OPTIMIZERS  -------
    # args.model_opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=args.momentum)
    args.model_opt = tf.keras.optimizers.Adam(args.lr)

    args.arch_opt = tf.keras.optimizers.Adam(args.arch_learning_rate, 0.5, 0.999)

    #  -------   LOSS FUNCTION  -------

    args.loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True) if CLASS_NUM == 1 else tf.keras.losses.CategoricalCrossentropy(from_logits=True)


def main():
    if not tf.test.is_gpu_available():
        raise Warning('Tensorflow not using GPU!')

    prepare()
    print('Starting Architecture Search')
    genotype = architecture_search()
    print('Architecture Search Finished \n Genotype: {} \n Starting Discreet Model Training'.format(genotype))
    train_discreet_model(genotype)
    print(
        'Finished Discreet Model Training \n Logs and trained model saved at {}'.format(args.output_dir))


DATA_PATH = './../../data/xtrains_dataset'
DATASET = 'XTRAINS'
VAL_PORTION = 0.1
TEST_PORTION = 0.5
DATASET_SIZE = 2000
EPOCHS = 100
OUTPUT_DIR = "./outputs/"
EXTRACTOR_PATH = './extractors/xtrains_typeA'
LAYERS = 1

if __name__ == "__main__":
    args = parse_args()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    CLASS_NUM = len(args.sec_labels)
    main()
