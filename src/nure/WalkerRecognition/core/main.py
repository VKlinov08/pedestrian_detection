from liblinear.liblinearutil import *
from src.nure.WalkerRecognition.utils import *
import pathlib

path_depth = len(pathlib.Path('.').resolve( ).parents)
ROOT_PATH = pathlib.Path('.').resolve( ).parents[path_depth - 4]
RESOURCES_PATH = sorted(ROOT_PATH.glob('**/resources'))
if len(RESOURCES_PATH) == 0:
    raise FileNotFoundError("Train directory not found!")

TRAIN_IMAGES_PATH = RESOURCES_PATH[0] / 'train'
TRAIN_Y_PATH = TRAIN_IMAGES_PATH / 'train-processed.idl'
TEST_IMAGES_PATH = RESOURCES_PATH[0] / 'test-public'
TEST_Y_PATH = TEST_IMAGES_PATH / 'test-processed.idl'
SAVE_MODEL_PATH = RESOURCES_PATH[0] / 'svm.model'

if __name__ == '__main__':
    train_images, train_names = load_images(TRAIN_IMAGES_PATH, with_names=True)
    train_labels = load_labels_from_idl(TRAIN_Y_PATH)
    test_images, test_names = load_images(TEST_IMAGES_PATH, with_names=True)
    test_labels = load_labels_from_idl(TEST_Y_PATH)

    step = 15
    train_x, train_y = prepare_train_datasets(train_images, train_names, train_labels,
                                              setsize=400,
                                              overlapping=10)

    coordinates, test_x, test_y = prepare_test_datasets(test_images, test_names, test_labels,
                                                        window_step=step,
                                                        d_thresh=15)
    # Подготавливаем тестовую выборку - извлекаем патчи, из патчей hog
    # Создаём модель
    prob = problem(train_y, train_x)
    param = parameter('-s 3')
    # Обучаем модель
    m = train(prob, param)
    # save_model(str(SAVE_MODEL_PATH), m)
    # m = load_model(str(SAVE_MODEL_PATH))

    # Валидируем модель
    predicted_labels = []
    # testing_indices = np.concatenate((np.arange(0, 25), np.arange(60, 80)))
    testing_indices = np.arange(0, len(test_images), 4)
    for i in testing_indices:
        p_label, p_acc, p_val = predict(test_y[i], test_x[i], m)
        predicted_labels.append(p_label)
        print(p_label, p_acc, p_val, sep='\n')


    # Функция детектированя по меткам и координатам и отрисовки рамочки
    for i, j in zip(testing_indices, range(len(testing_indices))):
        detect(predicted_labels[j], coordinates[i], test_images[i], step*3.5)
    # Функция подсчёта точности и полноты
    # precision, recall = composite_accuracy(test_y[testing_indices],
    # predicted_labels, test_images[testing_indices], coordinates[testing_indices])

    pass


