
from src.data_preparation.prepare_data import get_data
from src.model_training.train_test_model import run_train_test


#architectures = ["vgg16", "densenet121", "inceptionv3", "mobilenet", "resnet101", "xception"]

def grid_search():
    for pic_shape in [(32,32), (64,64), (100,100)]:
        for aug in [True, False]:
            for gray in [True, False]:
                all_data, two_cols_data = get_data(pic_shape, 100000, gray=gray)
                epoch = 30
                for batch_size in [16,32]:
                    for model_type in [1,2]:
                        for optimizer in ["adam", "sgd"]:
                            for data in [(two_cols_data, "two"), (all_data, "all")]:
                                run_train_test("grid_cnn",gray, aug, pic_shape, False, epoch, batch_size, model_type, optimizer, data)


if __name__ == "__main__":
    grid_search()

