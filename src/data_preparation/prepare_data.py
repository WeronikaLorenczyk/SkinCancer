from datetime import datetime

import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import pickle
from src.data_preparation.my_data_generator import DataGenerator
import numpy as np


def split(data, train_percent):
    #sepret division for each ilness
    train = []
    test = []
    for i in range(7):
        part_data = data[data["dx_id"] == i]
        part_data = part_data.sample(frac=1)
        train.append(part_data[:int(train_percent * len(part_data))])
        test.append(part_data[int(train_percent * len(part_data)):])

    return pd.concat(train).sample(frac=1), pd.concat(test).sample(frac=1)


def get_data(pic_shape, leng=100000, gray=False):
    #read data

    metadata = pd.read_csv("../../data/HAM10000_metadata.csv")[:leng]
    pic_path = "../../data/ham10000_images/"
    if gray:
        pic_path = "../../data/gray/"

    metadata["path"] = metadata["image_id"].map(lambda x : pic_path + x + ".jpg")
    metadata["image"] = metadata["path"].map(imread)

    dx_id_dict = {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}
    metadata["dx_id"] = metadata["dx"].map(dx_id_dict.get)

    #scale pictures
    metadata["image"] = metadata["image"].map(lambda pic: resize(pic, pic_shape))
    print("skalowanie", datetime.now())

    train, test = split(metadata[["image", "dx_id"]], 0.7)

    for i in range(7):
        print(i, len(train[train["dx_id"] == i]))



    all_train_gen = DataGenerator(np.asarray(list(train["image"])).reshape((len(train), pic_shape[0], pic_shape[1], 3)),
                                   np.asarray(list(train["dx_id"])).reshape((len(train))), 7)
    all_test_gen = DataGenerator(np.asarray(list(test["image"])).reshape((len(test), pic_shape[0], pic_shape[1], 3)),
                                 np.asarray(list(test["dx_id"])).reshape((len(test))), 7)

    class_weights = {i: len(train) / (len(train[train["dx_id"] == i]) * 7) for i in range(7)}

    all_data = all_train_gen, all_test_gen, class_weights

    train = train[(train["dx_id"] == 1) | (train["dx_id"] == 3)]
    test = test[(test["dx_id"] == 1) | (test["dx_id"] == 3)]
    test["dx_id"] = [0 if i == 1 else 1 for i in test["dx_id"]]
    train["dx_id"] = [0 if i == 1 else 1 for i in train["dx_id"]]

    class_weights = {0: len(train)/(len(train[train["dx_id"] == 0])*2), 1: len(train)/(len(train[train["dx_id"] == 1])*2)}

    two_train_gen = DataGenerator(np.asarray(list(train["image"])).reshape((len(train), pic_shape[0], pic_shape[1], 3)),
                                  np.asarray(list(train["dx_id"])).reshape((len(train))), 2)
    two_test_gen = DataGenerator(np.asarray(list(test["image"])).reshape((len(test), pic_shape[0], pic_shape[1], 3)),
                                 np.asarray(list(test["dx_id"])).reshape((len(test))), 2)

    two_cols_data = two_train_gen, two_test_gen, class_weights

    res = (all_data, two_cols_data)
    #with open("data/data32.pickle", "wb") as f:
    #    pickle.dump(res, f)
    return res


if __name__ == "__main__":
    res = get_data((32,32), 1000000, gray=False)
    with open("../../data/data32.pickle", "wb") as f:
        pickle.dump(res, f)

