from tensorflow.keras import layers, models, losses, callbacks
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.resnet import ResNet101
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.xception import Xception


def my_model (model_type, shape):
    if model_type == 1:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(shape[0], shape[1], 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        return model
    if model_type == 2:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(shape[0], shape[1], 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        if shape[0]>40:
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        return model


def transfer_model(model_name, input_shape, classes_nr):
    new_input = Input(shape=(input_shape[0], input_shape[1], 3))

    if model_name == "vgg16":
        model = VGG16(include_top=False, input_tensor=new_input)
    if model_name == "densenet121":
        model = DenseNet121(include_top=False, input_tensor=new_input)
    if model_name == "inceptionv3":
        model = InceptionV3(include_top=False, input_tensor=new_input)
    if model_name == "mobilenet":
        model = MobileNet(include_top=False, input_tensor=new_input)
    if model_name == "resnet101":
        model = ResNet101(include_top=False, input_tensor=new_input)
    if model_name == "xception":
        model = Xception(include_top=False, input_tensor=new_input)

    for layer in model.layers:
        layer.trainable = False
    flat1 = layers.Flatten()(model.layers[-1].output)
    class1 = layers.Dense(1024, activation='relu')(flat1)
    class2 = layers.Dense(256, activation='relu')(class1)
    output = layers.Dense(classes_nr, activation='softmax')(class2)
    model = Model(inputs=model.inputs, outputs=output)
    return model