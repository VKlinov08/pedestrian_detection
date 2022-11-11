import cv2
import pathlib
import numpy as np


def load_images(images_path: pathlib.Path, flag=cv2.IMREAD_COLOR, suffix='.png',
                with_names=False):
    train_images = []
    names = []
    path_patterns = sorted(images_path.glob(f'*{suffix}'),
                           key=lambda path: int(path.stem))
    for path in path_patterns:
        str_path = str(path)
        img = cv2.imread(str_path, flag)
        train_images.append(img)

    if with_names:
        for path in path_patterns:
            img_name = path.stem
            names.append(img_name)

    if with_names:
        return train_images, np.asarray(names, dtype=object)
    return train_images


def get_hog(images):
    # Скорректировать параметры под текущую задачу
    winSize = (80, 200)
    blockSize = (8, 8)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, 1, -1, 0, 0.2, 1, 64, True)

    histograms = []
    for image in images:
        hist = hog.compute(image)
        histograms.append(hist)

    return histograms


def load_labels_from_idl(load_path: pathlib.Path, encoding='ASCII'):
    path = pathlib.Path(load_path)
    if not path.exists( ):
        raise FileExistsError("Path doesn't exist!")
    if not path.is_file( ):
        raise FileNotFoundError("Path doesn't point to a file!")

    samples = [sample.split('\t') for sample in path.read_text(encoding).split('\n') if sample != '']
    dataset = [{'id': int(sample[0]),
                'y0': int(sample[1]),
                'x0': int(sample[2]),
                'y1': int(sample[3]),
                'x1': int(sample[4])
                } for sample in samples]
    return dataset


def crop_image(image, label):
    y0, x0, y1, x1 = list(label.values( ))[1:]
    return image[y0:y1, x0:x1]


def unique_ids_from_labels(labels):
    label_values = [list(sample.values( )) for sample in labels]
    label_values_T = np.array(label_values).T
    unique_label_ids = np.unique(label_values_T[0])
    return unique_label_ids


def create_y_by_images(images_names, labels):
    unique_ids = set(unique_ids_from_labels(labels))

    def binary_labeling(name):
        if int(name) in unique_ids:
            return 1
        else:
            return -1

    y = list(map(binary_labeling, images_names))
    return y


def crop_images(images, labels, names, with_y=False):
    cropped_images = []
    name_pointer = 0
    if not with_y:
        for label in labels:
            while label['id'] != int(names[name_pointer]):
                name_pointer += 1
            new_image = crop_image(images[name_pointer], label)
            cropped_images.append(new_image)
        return cropped_images
    else:
        y = []
        unique_ids = set(unique_ids_from_labels(labels))
        for label in labels:
            # Current image is a background and it is missing from the .idl labels.
            # Its class label is -1 and we can choose a random window.
            while True:
                if int(names[name_pointer]) not in unique_ids:
                    f_label = {'id': names[name_pointer],
                               'y0': 0,
                               'x0': 0,
                               'y1': 200,
                               'x1': 80}
                    new_image = crop_image(images[name_pointer], f_label)
                    cropped_images.append(new_image)
                    y.append(-1)
                    name_pointer += 1
                    continue
                # We can get multiple labels (2+ pedestriand ) on a single image,
                # so we need to check it.
                if label['id'] == int(names[name_pointer]):
                    new_image = crop_image(images[name_pointer], label)
                    cropped_images.append(new_image)
                    y.append(1)
                    break
                else:
                    name_pointer += 1
        return cropped_images, y


def cut_background(image, label, patch_width, max_overlapping=30):
    max_width = image.shape[1]
    # Loop until we get a position pretty far from a pedestrian.
    while True:
        x1_rand = np.random.randint(patch_width, max_width)

        if patch_width - abs(label['x1'] - x1_rand) > max_overlapping:
            continue
        x1 = x1_rand
        x0 = x1 - patch_width
        y1 = 200
        y0 = 0
        return image[y0:y1, x0:x1]


def cut_backgrounds(input_images, true_labels, size, patch_width, max_overlapping=30):
    background_images = []
    for _ in np.arange(size):
        rand_image_index = np.random.randint(0, len(input_images))
        new_background = cut_background(input_images[rand_image_index],
                                        true_labels[rand_image_index],
                                        patch_width,
                                        max_overlapping)
        background_images.append(new_background)
    return background_images


def prepare_train_datasets(train_images, train_names, train_labels,
                           patch_width=80, setsize=250, overlapping=30):
    walkers_patches = crop_images(train_images, train_labels, train_names)
    walkers_descriptors = get_hog(walkers_patches)
    walker_labels = create_y_by_images(train_names, train_labels)

    background_images = cut_backgrounds(train_images, train_labels, setsize, patch_width, overlapping)
    background_descriptors = get_hog(background_images)
    background_labels = [-1] * setsize

    train_hogs = walkers_descriptors + background_descriptors
    train_y = np.asarray(walker_labels + background_labels)
    return np.asarray(train_hogs), train_y


# def prepare_test_datasets(test_images, image_names, test_labels, window_size=(80, 200), window_step=10):
#     coordinates_list = []
#     test_x = []
#     for image in test_images:
#         coordinates, descriptors = slide_extract(image, window_size=window_size, step=window_step)
#         test_x.extend(descriptors)
#         coordinates_list.append(coordinates)
#
#     test_y = create_y_by_images(image_names, test_labels)
#     test_y_by_windows = set_labels_to_windows(test_y, test_labels, coordinates_list, image_names)
#     return coordinates_list, test_x, test_y_by_windows


def slide_extract(image, window_size=(80, 200), step=12):
    hIm, wIm = image.shape[:2]
    windows = []
    w1_range = np.arange(0, wIm - window_size[0], step)
    w2_range = np.arange(window_size[0], wIm, step)
    width_pairs = np.column_stack((w1_range, w2_range))
    for w1, w2 in width_pairs:
        windows.append(image[0:200, w1:w2])

    coords = np.array([(w1, w2, 0, 200) for w1, w2 in width_pairs])
    features = get_hog(windows)
    return coords, np.asarray(features)


def get_labels_by_name(name, labels):
    return list(filter(lambda label: label['id'] == int(name),
                       labels))


def set_labels_to_windows(image_y, labels, windows_coordinates, names, distance_threshold=10):
    def calc_distances(current_point, destination_points):
        num_points = len(destination_points)
        points1 = np.full((num_points,), current_point)
        points2 = np.array(destination_points)
        result_distances = np.abs(np.subtract(points1, points2))
        return result_distances

    labels_to_windows = []
    for i, y in enumerate(image_y):
        if y == -1:
            size = len(windows_coordinates[i])
            windows_y = [-1] * size
            labels_to_windows.extend(windows_y)
        else:
            windows_y = []
            name = names[i]
            # Retrieve labels which have id == int(name)
            current_labels = get_labels_by_name(name, labels)

            for coords in windows_coordinates[i]:
                x0 = coords[0]
                destination_x = list(map(lambda label: label['x0'], current_labels))
                distances = calc_distances(x0, destination_x)
                window_y = 1 if any(distances <= distance_threshold) else -1
                windows_y.append(window_y)
            labels_to_windows.extend(windows_y)

    return labels_to_windows


def prepare_test_datasets(test_images, image_names, test_labels,
                          window_size=(80, 200), window_step=10, d_thresh=10):
    coordinates_list = []
    test_x = []
    for image in test_images:
        coordinates, descriptors = slide_extract(image, window_size=window_size, step=window_step)
        test_x.append(descriptors)
        coordinates_list.append(coordinates)

    test_y = create_y_by_images(image_names, test_labels)
    test_y_by_windows = []
    # Set label = +1 to windows that are pretty close to a pedestrian an -1 otherwise.
    for i, y in enumerate(test_y):
        windows_y = set_labels_to_windows([y],
                                          test_labels,
                                          [coordinates_list[i]],
                                          [image_names[i]],
                                          distance_threshold=d_thresh)
        test_y_by_windows.append(windows_y)
    return np.asarray(coordinates_list, dtype=object), np.asarray(test_x), np.asarray(test_y_by_windows)


def get_overlapping_ratio(prediction_window, true_window):
    width = prediction_window[1] - prediction_window[0]
    overlapping = width - abs(prediction_window[0] - true_window[0])
    return overlapping / width


def count_matching(predicted_detections, true_detections):
    if len(true_detections) == 0 and len(predicted_detections) != 0:
        return (0, len(predicted_detections), 0)
    pedestrian_matches = [False] * len(true_detections)
    TP_det = 0
    FP = 0
    for prediction in predicted_detections:
        true_positive_matches = []
        for i, true_det in enumerate(true_detections):
            if get_overlapping_ratio(prediction, true_det) >= 0.5:
                true_positive_matches.append(True)
                TP_det += 1
                if not pedestrian_matches[i]:
                    pedestrian_matches[i] = True
            else:
                true_positive_matches.append(False)
        if not any(true_positive_matches):
            FP += 1
    TP = sum(pedestrian_matches)
    return TP_det, FP, TP


# Подсчитывает значение метки изображения по меткам каждого окна
# и подсчитывает точность
def composite_evaluation(predicted_detections,
                         true_labels):
    true_positive_det = 0.0
    general_true = float(len(true_labels))
    false_positive = 0.0
    true_positive_ped = 0.0

    for i, current_detections in enumerate(predicted_detections):
        true_current_labels = true_labels[i]
        true_detections = [(label['x0'],
                            label['x1'],
                            label['y0'],
                            label['y1']) for label in true_current_labels]

        TP_det, FP, TP = count_matching(current_detections, true_detections)
        true_positive_det += TP_det
        false_positive += FP
        true_positive_ped += TP

    precision = true_positive_ped / general_true
    recall = true_positive_det / (true_positive_det + false_positive)
    return precision, recall


def cluster_windows(windows, threshold):
    if len(windows) <= 1:
        return windows
    clusters = []
    x1 = windows[:, 0]
    distances = np.abs(np.subtract(x1[1:], x1[:-1]))
    i = 0
    current_cluster = [i]
    while i < len(windows) - 1:
        if distances[i] < threshold:
            current_cluster.append(i + 1)
        else:
            clusters.append(current_cluster)
            current_cluster = [i + 1]
        i += 1
    clusters.append(current_cluster)
    return clusters


def compress_clusters(boxes, clusters):
    if boxes.shape[0] <= 1 or boxes.shape[1] <= 1:
        return boxes
    compressed_boxes = []
    for cluster in clusters:
        center = np.median(boxes[cluster], axis=0).astype(np.int32)
        compressed_boxes.append(center)
    return compressed_boxes


def detect(windows_predictions, windows_coordinates, overlap_threshold=0.2):
    all_windows = np.asarray(windows_coordinates)
    positive_windows = all_windows[list(map(lambda y: y == 1.0, windows_predictions))]
    if len(positive_windows) == 0:
        return []

    clusters = cluster_windows(positive_windows, overlap_threshold)
    chosen_windows = compress_clusters(positive_windows, clusters)
    return chosen_windows


def show_images_with_boxes(windows_coordinates, image):
    boxes = np.asarray(windows_coordinates)

    bounded_image = image.copy( )
    for (startX, endX, startY, endY) in boxes:
        cv2.rectangle(bounded_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imshow("Original", image)
    cv2.imshow("Windows minimizing", bounded_image)
    cv2.waitKey(0)
