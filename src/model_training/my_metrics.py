
import numpy as np
import keras
import sklearn.metrics as sklm
from keras import utils

class MetricsTwo(keras.callbacks.Callback):

    def __init__(self, validation_data, class_weights):
        super().__init__()
        self.validation_data = validation_data
        self.class_weights = class_weights

        self.sensitivity = []
        self.specificity = []
        self.auc = []
        self.tp = []
        self.tn = []
        self.fp = []
        self.fn = []
        self.w_acc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data.x_data))
        targ = self.validation_data.y_data

        self.auc.append(sklm.roc_auc_score(targ, score[:, 1]))

        predict = np.argmax(score, axis=1)
        (tn, fp), (fn, tp) = sklm.confusion_matrix(targ, predict, normalize="true")

        print("to jaaa",tn, fp, fn, tp, self.auc[-1])

        self.specificity.append(tn/(fp+tn))
        self.sensitivity.append(tp/(tp+fn))
        self.tn.append(tn)
        self.tp.append(tp)
        self.fn.append(fn)
        self.fp.append(fp)
        self.w_acc.append((tp+tn)/(tp+tn+fp+fn))
        return


class MetricsAll(keras.callbacks.Callback):

    def __init__(self, validation_data, class_weights):
        super().__init__()
        self.validation_data = validation_data
        self.class_weights = class_weights

        self.w_acc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data.x_data))
        targ = self.validation_data.y_data
        predict = np.argmax(score, axis=1)

        conf = sklm.confusion_matrix(targ, predict, normalize="true")
        good = sum(conf.diagonal())
        acc = good / sum(sum(conf))
        self.w_acc.append(acc)
        print("weight", acc, good, sum(sum(conf)))
        return
