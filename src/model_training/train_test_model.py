
from tensorflow.keras import layers, losses, callbacks
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import pickle
from src.model_training.models import my_model, transfer_model
from src.model_training.my_metrics import MetricsTwo, MetricsAll
import tensorflow as tf
import numpy as np
from src.analysis.plots import plot_roc, plot_confusion_matrix
from src.data_preparation.my_data_generator import DataGenerator


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


def run_train_test(test_name,gray, augmentation, pic_shape,if_transfer, epoch,batch_size,model_type,optimizer,data):
    data_string = f"pic_shape{pic_shape}_aug={augmentation}_gray={gray}"

    info_str = f"{test_name}/optimizer={optimizer}_data={data[1]}_epoch={epoch}_batch={batch_size}_model={model_type}_data_string={data_string}"
    logdir = "../../logs/"+info_str
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

    train_gen, test_gen, class_weights = data[0]

    train_gen = DataGenerator(train_gen.x_data, train_gen.y_data, train_gen.class_number)
    test_gen = DataGenerator(test_gen.x_data, test_gen.y_data, test_gen.class_number)

    train_gen.set_parameters(batch_size, augmentation)
    test_gen.set_parameters(batch_size, False)
    print("class weights ",class_weights)

    if if_transfer:
        model = transfer_model(model_type, pic_shape, train_gen.class_number)
    else:
        model = my_model(model_type, pic_shape)
        model.add(layers.Dense(train_gen.class_number, activation="softmax"))

    if data[1] == "two":
        metrics = MetricsTwo(test_gen, class_weights)
    else:
        metrics = MetricsAll(test_gen, class_weights)

    model.compile(optimizer=optimizer,
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    if augmentation:
        class_weights = None

    model.fit(train_gen, epochs=epoch, batch_size=batch_size,
                        validation_data=test_gen, class_weight=class_weights,
                        callbacks=[tensorboard_callback, metrics, learning_rate_reduction])

    pred = model.predict(test_gen.x_data)
    pred_classes = np.argmax(pred, axis=1)
    true_classes = test_gen.y_data
    plot_confusion_matrix(true_classes, pred_classes, data[1], "../../diagrams/conf/"+info_str+".png")
    add_tb_info(metrics,epoch, data[1], logdir)
    if data[1] == "two":
        plot_roc(test_gen.y_data, pred[:,1], "../../diagrams/roc/"+info_str+".png")


if __name__ == "__main__":
    with open("../../data/data32.pickle", "rb") as f:
        all_data, two_cols_data = pickle.load(f)
    #
    run_train_test("stupid_test",False, True, (32, 32), False, 30, 16, 1, "adam", (two_cols_data, "two"))
    #
    # architectures = ["vgg16","densenet121","inceptionv3","mobilenet","resnet101","xception"]
    #grid_search()




