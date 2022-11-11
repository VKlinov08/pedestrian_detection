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
    all_test_labels = load_labels_from_idl(TEST_Y_PATH)

    step = 15
    train_x, train_y = prepare_train_datasets(train_images, train_names, train_labels,
                                              setsize=400,
                                              overlapping=10)

    windows_coordinates, test_x, test_y = prepare_test_datasets(test_images, test_names, all_test_labels,
                                                                window_step=step,
                                                                d_thresh=15)
    # Creating of a model
    prob = problem(train_y, train_x)
    param = parameter('-s 3')
    # Fitting of the model
    m = train(prob, param)
    # save_model(str(SAVE_MODEL_PATH), m)
    # m = load_model(str(SAVE_MODEL_PATH))

    # Testing the model
    predicted_y = []
    testing_indices = np.arange(0, len(test_images), 4)
    for i in testing_indices:
        p_label, p_acc, p_val = predict(test_y[i], test_x[i], m)
        predicted_y.append(p_label)
        print(p_label, p_acc, p_val, sep='\n')

    image_detections = []
    # Getting windows with detected pedestrians
    for i, j in zip(testing_indices, range(len(testing_indices))):
        current_detections = detect(predicted_y[j], windows_coordinates[i], step * 3.5)
        show_images_with_boxes(current_detections, test_images[i])
        image_detections.append(current_detections)

    # Evaluating results
    current_true_labels = [get_labels_by_name(name, all_test_labels) for name in test_names[testing_indices]]
    precision, recall = composite_evaluation(image_detections, test_names[testing_indices], current_true_labels)

    pass


