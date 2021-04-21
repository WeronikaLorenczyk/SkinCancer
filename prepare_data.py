import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle


def standarize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img


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


def one_hot_enc(data):
    result = np.zeros((len(data), 7))
    for i in range(len(data)):
        result[i][int(data.iloc[i])] = 1
    return result


def get_data(pic_shape, leng):
    #read data
    metadata = pd.read_csv("data/HAM10000_metadata.csv")[:leng]
    metadata["path"] = metadata["image_id"].map(lambda x : "data/ham10000_images/"+ x + ".jpg")
    metadata["image"] = metadata["path"].map(imread)

    dx_id_dict = {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}

    metadata["dx_id"] = metadata["dx"].map(dx_id_dict.get)
    class_weights = {i: 1/len(metadata[metadata["dx_id"] == i]) for i in range(7)}

    #scale and normalise pictures
    metadata["image"] = metadata["image"].map(lambda pic : resize(pic, pic_shape))
    #here'll be some other transformations
    metadata["image"] = metadata["image"].map(standarize)

    train, test = split(metadata[["image", "dx_id"]], 0.7)

    all_data = (np.asarray(list(train["image"])).reshape(((len(train)), pic_shape[0], pic_shape[1], 3)), one_hot_enc(train["dx_id"]), np.asarray(list(test["image"])).reshape((len(test), pic_shape[0], pic_shape[1], 3)), one_hot_enc(test["dx_id"]), class_weights)

    train = train[(train["dx_id"] == 1) | (train["dx_id"] == 3)]
    test = test[(test["dx_id"] == 1) | (test["dx_id"] == 3)]
    class_weights = {0: 1/class_weights[1], 1: 1/class_weights[3]}
    two_cols_data = (np.asarray(list(train["image"])).reshape(((len(train)), pic_shape[0], pic_shape[1], 3)), np.asarray([1 if el ==3 else 0 for el in train["dx_id"]]).reshape((len(train), 1)), np.asarray(list(test["image"])).reshape((len(test), pic_shape[0], pic_shape[1], 3)), np.asarray([1 if el ==3 else 0 for el in test["dx_id"]]).reshape((len(test), 1)), class_weights)

    return (all_data, two_cols_data)


if __name__ == "__main__":
    res = get_data((32,32), 100000)
    with open("data/data32.pickle", "wb") as f:
        pickle.dump(res, f)