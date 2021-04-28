from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras import layers, models, losses, callbacks
import numpy as np
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from plots import plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
import keras
import sklearn.metrics as sklm
import tensorflow as tf
from prepare_data import get_data
from models import my_model, transfer_model


class MetricsTwo(keras.callbacks.Callback):

    #TODO tu rowne klasy dla auc
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
        score = np.asarray(self.model.predict(self.validation_data[0]))
        targ = np.argmax(self.validation_data[1], axis=1)

        self.auc.append(sklm.roc_auc_score(targ, score[:,1]))

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
        score = np.asarray(self.model.predict(self.validation_data[0]))
        targ = np.argmax(self.validation_data[1], axis=1)
        predict = np.argmax(score, axis=1)

        conf = sklm.confusion_matrix(targ, predict, normalize="true")
        good = sum(conf.diagonal())
        acc =good / sum(sum(conf))
        self.w_acc.append(acc)
        print("weight",acc, good, sum(sum(conf)))
        return


def plot_conf_matrix(Y_pred,test_y, data_type, info_str ):
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(test_y, axis=1)
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

    # plot the confusion matrix
    if data_type == "all":
        classes=('bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec')

    else:
        classes = ('mel','nv')
    plot_confusion_matrix(confusion_mtx, classes=classes)
    plt.savefig("diagrams/test3/conf/" + info_str + ".png")
    plt.clf()


def plot_roc(test_y, Y_pred, info_str):
    fpr, tpr, _ = roc_curve(test_y[:, 1], Y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("diagrams/test3/roc/" + info_str + ".png")
    plt.clf()

def add_tb_info(metrics, epoch, data_type, logdir):
    summary_writer = tf.summary.create_file_writer(logdir+"/validation")
    with summary_writer.as_default():
        if data_type == "two":
            for i in range(epoch):
                tf.summary.scalar('sensitivity', metrics.sensitivity[i], step=i)
                tf.summary.scalar('specificity', metrics.specificity[i], step=i)
                tf.summary.scalar('auc', metrics.auc[i], step=i)
                tf.summary.scalar('tp', metrics.tp[i], step=i)
                tf.summary.scalar('tn', metrics.tn[i], step=i)
                tf.summary.scalar('fp', metrics.fp[i], step=i)
                tf.summary.scalar('fn', metrics.fn[i], step=i)
                tf.summary.scalar('weighted acc', metrics.w_acc[i], step=i)
        else:
            for i in range(epoch):
                tf.summary.scalar('weighted acc', metrics.w_acc[i], step=i)


def run_train_test(data_string, pic_shape,if_transfer, epoch,batch_size,model_type,optimizer,data):

    info_str = f"optimizer={optimizer}_data={data[1]}_epoch={epoch}_batch={batch_size}_model={model_type}_data_string={data_string}"
    logdir = "logs/test3/"+info_str
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

    train_X, train_y, test_X, test_y, class_weights_test, class_weights_train = data[0]

    if "aug=True" in data_string:
        train_datagen = ImageDataGenerator(rotation_range=10,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           fill_mode='nearest')

    else:
        train_datagen = ImageDataGenerator(rescale=1)

    train_datagen.fit(train_X)

    test_datagen = ImageDataGenerator(rescale=1)
    test_datagen.fit(test_X)
    train_data = train_datagen.flow(train_X, train_y, batch_size=batch_size)
    test_data = test_datagen.flow(test_X, test_y, batch_size=batch_size)

    print("class weights ",class_weights_train, class_weights_test)

    print(train_X.shape, test_X.shape, info_str)

    if if_transfer:
        model = transfer_model(model_type, pic_shape, test_y.shape[1])
    else:
        model = my_model(model_type, pic_shape)
        model.add(layers.Dense(test_y.shape[1], activation="softmax"))

    if data[1] == "two":
        metrics = MetricsTwo((test_X, test_y), class_weights_test)
    else:
        metrics = MetricsAll((test_X, test_y), class_weights_test)

    model.compile(optimizer=optimizer,
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    history = model.fit(train_data, epochs=epoch, batch_size=batch_size,
                        validation_data=test_data, class_weight=class_weights_train,
                        callbacks=[tensorboard_callback, metrics, learning_rate_reduction])

    Y_pred = model.predict(test_X)
    plot_conf_matrix(Y_pred,test_y, data[1], info_str)
    add_tb_info(metrics,epoch, data[1], logdir)
    weights = [class_weights_test[0] if i[0]==1 else class_weights_test[1] for i in test_y]
    plot_roc(test_y, Y_pred)



def grid_search():
    for pic_shape in [(32,32), (64,64), (100,100)]:
        for aug in [True]:
            all_data, two_cols_data = get_data(pic_shape, 100000, augmentation=aug, gray=not aug)
            data_string = f"pic_shape{pic_shape}_aug={aug}_gray={not aug}"
            for epoch in [15,30]:
                for batch_size in [16,32]:
                    for model_type in [1,2]:
                        for optimizer in ["adam", "sgd"]:
                            for data in [(two_cols_data, "two"), (all_data, "all")]:
                                run_train_test(data_string, pic_shape, False, epoch, batch_size, model_type, optimizer, data)


if __name__ == "__main__":
    with open("data/data32.pickle", "rb") as f:
        all_data, two_cols_data = pickle.load(f)
    #
    run_train_test("xx", (32, 32), False, 2, 16, 1, "adam", (two_cols_data, "two"))
    #
    # architectures = ["vgg16","densenet121","inceptionv3","mobilenet","resnet101","xception"]
    #grid_search()




