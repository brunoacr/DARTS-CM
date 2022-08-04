import os.path

import tensorflow as tf
import pandas as pd


class MetricsLogger(tf.keras.callbacks.Callback):

    def __init__(self, output_dir, name):
        super(MetricsLogger, self).__init__()
        self.name = name
        self.metrics = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
        self.output_dir = output_dir
        # self.genotype_record_file = open(output_dir + name + "_genotype_record_file.txt", "w")
        # self.metrics_file = open(output_dir + name + "_metrics.txt", "w")

    def on_epoch_end(self, epoch, logs=None):
        # self.metrics_file.write("epoch: {}  |  loss: {}  |  acc: {}  |  val_loss: {}  |  val_acc: {}\n".format(
        #                             epoch, logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy']))
        # self.genotype_record_file.write('Epoch {} : {}\n'.format(epoch, self.model.get_genotype()))

        logs['epoch'] = epoch
        new_metrics_row = pd.DataFrame(logs, index=[epoch])
        self.metrics = pd.concat([self.metrics, new_metrics_row])

    def on_train_end(self, logs=None):
        self.metrics.to_csv(os.path.join(self.output_dir, self.name + '_metrics.txt'))


class GenotypeLogger(tf.keras.callbacks.Callback):

    def __init__(self, output_dir, name):
        super(GenotypeLogger, self).__init__()
        self.name = name
        self.genotype_record = pd.DataFrame(columns=['epoch', 'genotype'])
        self.output_dir = output_dir


    def on_epoch_end(self, epoch, logs=None):
        new_genotype_row = pd.DataFrame({'epoch': epoch, 'genotype': str(self.model.get_genotype())}, index=[epoch])
        self.genotype_record = pd.concat([self.genotype_record, new_genotype_row])

    def on_train_end(self, logs=None):
        self.genotype_record.to_csv(os.path.join(self.output_dir, self.name + '_genotype.txt'))
