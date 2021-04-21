from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models, losses, callbacks
from prepare_data import get_data
from datetime import datetime
import numpy as np
from plots import plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
import tensorflow.keras.backend as K


def my_model_all(model_type):
    if model_type == 1:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        return model
    if model_type == 2:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        return model


#{Sensitivity} - TP/(TP + FN)
# {Specificity} - TN/(FP + TN)

def sensitivity(y_true, y_pred):
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tn_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    tn = K.sum(K.cast(K.all(tn_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))
    print(tp,tn,fp,fn)
    return tp/(tp+fn)


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return tn/(fp+tn)


def grid_search_all(all_data, two_cols_data):

    for epoch in [15,30]:
        for batch_size in [16,32]:
            for model_type in [1,2]:
                for data in [(two_cols_data, "two"), (all_data, "all")]:

                    info_str = f"data={data[1]}epoch={epoch}_batch={batch_size}_model={model_type}"

                    logdir = "logs/test1/"+info_str
                    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

                    train_X, train_y, test_X, test_y, class_weights = data[0]
                    model = my_model_all(model_type)
                    model.add(layers.Dense(test_y.shape[1]))

                    if data[1] == "all":
                        metrics = ["accuracy"]
                    else:
                        metrics = ['accuracy', 'Recall', 'Precision']

                    model.compile(optimizer='adam',
                                  loss=losses.CategoricalCrossentropy(from_logits=True),
                                  metrics=metrics, weighted_metrics=metrics)

                    history = model.fit(train_X, train_y, epochs=epoch, batch_size=batch_size,
                                        validation_data=(test_X, test_y), class_weight=class_weights,  callbacks=[tensorboard_callback])

                    Y_pred = model.predict(test_X)
                    Y_pred_classes = np.argmax(Y_pred, axis=1)
                    Y_true = np.argmax(test_y, axis=1)
                    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

                    # plot the confusion matrix
                    if data[1] == "all":
                        classes=('bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec')
                    else:
                        classes = ('mel','nv')
                    plot_confusion_matrix(confusion_mtx, classes=classes)
                    plt.savefig("diagrams/test1/" + info_str + ".png")
                    plt.clf()


if __name__ == "__main__":
    with open("data/data32.pickle", "rb") as f:
        all_data, two_cols_data = pickle.load(f)


    grid_search_all(all_data, two_cols_data)


