from pathlib import Path
from zipfile import ZipFile

import numpy as np
import os
import tensorflow as tf
import pandas as pd
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing import image
import math

from utils.activations_extractor import ActivationsExtractor

DATA_ROOT = Path('./data')


def get_images(path, df):
    data = []

    for i in range(len(df)):
        img_id = df.iloc[i]['id']
        img_path = Path(path, str(img_id) + '.jpg')
        img = image.load_img(img_path)
        data.append(img)

    # concatenate all block images
    data = np.stack(data, axis=0)
    data = tf.keras.utils.normalize(data)
    return data


def split_and_get_imgs(neg, pos, portion, path, balance_class):
    if portion == 0:
        return [], [], neg, pos

    if portion > 0:
        assert portion < 1
        cut = math.floor((len(neg) + len(pos)) * portion)
        cut_neg = cut // 2
        cut_pos = cut_neg
        if cut % 2 != 0: cut_pos += 1
        y = pd.concat([neg[:cut_neg], pos[:cut_pos]], ignore_index=True)
        neg, pos = neg[cut_neg:], pos[cut_pos:]
    else:
        y = pd.concat([neg, pos], ignore_index=True)

    x = get_images(path, y)
    y = y[balance_class].to_numpy()
    y = y.reshape((y.shape[0], 1))
    return x, y, neg, pos


def loadDataset(complexity, balance_class, size, val_split=0.2, test_split=0.2, padding=False):
    data_path = Path(DATA_ROOT, 'C' + str(complexity))
    labels = Path(data_path, 'labels.csv')
    images = Path(data_path, 'images')
    if not os.path.isdir(data_path):
        print('Downloading dataset')
        images_zip = hf_hub_download(repo_type='dataset', repo_id="bruno-cotrim/arch-max",
                                     filename="images/c{}.zip".format(complexity))

        os.makedirs(images)

        with ZipFile(images_zip) as f:
            ZipFile.extractall(f, path=images)

        labels_path = hf_hub_download(repo_type='dataset', repo_id="bruno-cotrim/arch-max",
                                      filename="labels/c1/all.csv")

        df = pd.read_csv(labels_path, index_col=0)
        df.to_csv(labels)

    val_split = val_split / (1 - test_split)

    df = pd.read_csv(labels, index_col=0)

    # balance

    balance_pos = df[df[balance_class] == 1]
    balance_neg = df[df[balance_class] == 0]

    if size == -1:
        size = min(len(balance_neg), len(balance_pos)) * 2

    if len(balance_pos) < size // 2:
        if not padding:
            raise Exception('Not enough examples of ' + balance_class + ' == 1')
        else:
            while len(balance_pos) < size // 2:
                balance_pos = pd.concat([balance_pos, balance_pos])
    if len(balance_neg) < size // 2:
        if not padding:
            raise Exception('Not enough examples of ' + balance_class + ' == 0')
        else:
            while len(balance_neg) < size // 2:
                balance_neg = pd.concat([balance_neg, balance_neg])

    pos = balance_pos.iloc[:size // 2]
    neg = balance_neg.iloc[:size // 2]

    x_test, y_test, neg, pos = split_and_get_imgs(neg, pos, test_split, images, balance_class)

    x_val, y_val, neg, pos = split_and_get_imgs(neg, pos, val_split, images, balance_class)

    x_train, y_train, _, _ = split_and_get_imgs(neg, pos, -1, images, balance_class)

    images = {'train': x_train, 'valid': x_val if val_split > 0 else [], 'test': x_test if test_split > 0 else None}
    labels = {'train': y_train, 'valid': y_val if val_split > 0 else [], 'test': y_test if test_split > 0 else None}

    return images, labels


EXTRACTOR_ROOT = Path('./extractors')


def loadExtractor(complexity, main_network):
    extractor = Path(EXTRACTOR_ROOT, 'ExtC{}_{}'.format(complexity, main_network))

    if not os.path.isdir(extractor):
        print('Downloading Extractor')
        zip = hf_hub_download(repo_type='dataset', repo_id="bruno-cotrim/arch-max",
                              filename='extractors/ExtC{}_{}.zip'.format(complexity, main_network))

        with ZipFile(zip) as f:
            ZipFile.extractall(f, path=EXTRACTOR_ROOT)

    return ActivationsExtractor.load(extractor)


# loadExtractor(1, 'Commercial')