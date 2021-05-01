import random

import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras_preprocessing.image import flip_axis

from keras.preprocessing.image import apply_affine_transform


# TODO normalizacja

def transform_pic(pic):
    theta = np.random.uniform(-20, 20)
    ty = np.random.uniform(-0.2, 0.2)
    tx = np.random.uniform(-0.2, 0.2)
    shear = np.random.uniform(-0.2, 0.2)
    zy = np.random.uniform(0.8, 1)
    zx = np.random.uniform(0.8, 1)
    pic = apply_affine_transform(pic, theta=theta, tx=tx, ty=ty, shear=shear, zx=zx, zy=zy, channel_axis=2)
    if random.randint(0, 1):
        pic = flip_axis(pic, 0)
    if random.randint(0, 1):
        pic = flip_axis(pic, 1)
    return pic


def normalize(pic):
    mean = np.mean(pic)
    std = np.std(pic)
    pic = (pic-mean)/std
    return pic


class DataGenerator(utils.Sequence):

    def __init__(self, x_data, y_data, class_number):
        self.y_data = y_data
        x_data = normalize(x_data)
        self.x_data = x_data
        self.x_data_dict = {i: x_data[np.equal(self.y_data, i)] for i in range(class_number)}
        self.lens = [len(self.x_data_dict[i]) - 1 for i in range(class_number)]
        self.class_number = class_number
        self.batch_size = None
        self.augmentation = None

    def __len__(self):
        return int(np.floor(len(self.y_data) / self.batch_size))

    def __getitem__(self, index):

        if self.augmentation:
            class_idx = np.random.randint(0, self.class_number, self.batch_size)
            indexes = [random.randint(0, self.lens[class_idx[i]]) for i in range(self.batch_size)]

            x = np.array([transform_pic(self.x_data_dict[cl][k]) for cl, k in zip(class_idx, indexes)])
            y = utils.to_categorical(class_idx, num_classes=self.class_number)

        else:
            indexes = range(len(self.x_data))[index * self.batch_size:(index + 1) * self.batch_size]
            x = np.array( self.x_data[indexes])
            y = utils.to_categorical(np.array(self.y_data[indexes]), num_classes=self.class_number)

        return x, y

    def set_parameters(self, batch_size, augmentation):
        self.batch_size = batch_size
        self.augmentation = augmentation

    def on_epoch_end(self):
        if not self.augmentation:
            print("pomieszane")
            p = np.random.permutation(len(self.x_data))
            self.x_data = self.x_data[p]
            self.y_data = self.y_data[p]
