import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle
from preprocessing import augmentation_func


def standarize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img


def split_and_multiply(data, train_percent, multi_list=None):
    #sepret division for each ilness

    if multi_list is None:
        multi_list = [1] * 7
    train = []
    test = []
    for i in range(7):
        part_data = data[data["dx_id"] == i]
        part_data = part_data.sample(frac=1)
        train.append(pd.concat([part_data[:int(train_percent * len(part_data))]]*multi_list[i]))
        test.append(part_data[int(train_percent * len(part_data)):])

    return pd.concat(train).sample(frac=1), pd.concat(test).sample(frac=1)


def one_hot_enc(data, cols=7):
    result = np.zeros((len(data), cols))
    for i in range(len(data)):
        result[i][int(data.iloc[i])] = 1
    return result


def get_data(pic_shape, leng=100000, augmentation=False, gray=False):
    #read data
    metadata = pd.read_csv("data/HAM10000_metadata.csv")[:leng]
    pic_path = "data/ham10000_images/"
    if gray:
        pic_path = "data/gray/"
    metadata["path"] = metadata["image_id"].map(lambda x : pic_path + x + ".jpg")
    metadata["image"] = metadata["path"].map(imread)

    dx_id_dict = {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}

    metadata["dx_id"] = metadata["dx"].map(dx_id_dict.get)


    #scale pictures
    metadata["image"] = metadata["image"].map(lambda pic : resize(pic, pic_shape))


    multi_list = [1]*7
    if augmentation:
        multi_list = [min(max(1,int(4000/max(len(metadata[metadata["dx_id"] == i]),1))),20)for i in range(7)]
        print(multi_list)

    train, test = split_and_multiply(metadata[["image", "dx_id"]], 0.7, multi_list)
    for i in range(7):
        print(i, len(train[train["dx_id"] == i]))

    class_weights_train = {i: len(train) / max(len(train[train["dx_id"] == i]), 1) for i in range(7)}
    class_weights_test = {i: len(test) / max(len(test[test["dx_id"] == i]), 1) for i in range(7)}

    if augmentation:
        train["image"] = train["image"].map(augmentation_func)

    # here'll be some other transformations
    train["image"] = train["image"].map(standarize)
    test["image"] = test["image"].map(standarize)

    all_data = (np.asarray(list(train["image"])).reshape(((len(train)), pic_shape[0], pic_shape[1], 3)), one_hot_enc(train["dx_id"]), np.asarray(list(test["image"])).reshape((len(test), pic_shape[0], pic_shape[1], 3)), one_hot_enc(test["dx_id"]), class_weights_test, class_weights_train)

    train = train[(train["dx_id"] == 1) | (train["dx_id"] == 3)]
    test = test[(test["dx_id"] == 1) | (test["dx_id"] == 3)]
    class_weights_test = {0: len(test)/max(len(test[test["dx_id"] == 1]),1), 1: len(test)/max(len(test[test["dx_id"] == 3]),1)}
    class_weights_train = {0: len(train) / max(len(train[train["dx_id"] == 1]),1), 1: len(train) / max(len(train[train["dx_id"] == 3]),1)}
    two_cols_data = (np.asarray(list(train["image"])).reshape(((len(train)), pic_shape[0], pic_shape[1], 3)), one_hot_enc(pd.DataFrame([1 if el ==3 else 0 for el in train["dx_id"]]), cols=2), np.asarray(list(test["image"])).reshape((len(test), pic_shape[0], pic_shape[1], 3)), one_hot_enc(pd.DataFrame([1 if el ==3 else 0 for el in test["dx_id"]]),cols=2), class_weights_test, class_weights_train)
    res = (all_data, two_cols_data)
    with open("data/data32.pickle", "wb") as f:
        pickle.dump(res, f)
    return all_data, two_cols_data


if __name__ == "__main__":
    res = get_data((32,32), 1000000, augmentation=True)
    with open("data/data32.pickle", "wb") as f:
        pickle.dump(res, f)

