import numpy as np
import cv2

import pathlib

path_depth = len(pathlib.Path('.').resolve( ).parents)
ROOT_PATH = pathlib.Path('.').resolve( ).parents[path_depth - 4]
RESOURCES_PATH = sorted(ROOT_PATH.glob('**/resources'))

points = np.array([[1.0, 2.1], [1, -1], [2, 3], [2, 1]], dtype=np.float32)
labels = np.array([0, 1, 0, 1], dtype=np.float32)
svm = cv2.ml.SVM_create()
svm.setGamma(1)
svm.setC(1)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.train(points, cv2.ml.ROW_SAMPLE, labels)

save_path = str(RESOURCES_PATH[0] / 'cv2svm.xml')
svm.save(save_path)

svm_loaded = cv2.SVM()
svm_loaded.load(save_path)

# import numpy as np
# import pandas as pd
# import os
# import time
# import random
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# from glob import glob
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score,confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from skimage.feature import hog
# 
# import PIL
# import cv2
# import pickle
# 
# car_paths = glob("../input/the-car-connection-picture-dataset" + "/*")[:5000]
# neg_paths = []
# 
# for class_path in glob("../input/natural-images/natural_images" + "/*"):
#     if class_path != "../input/natural-images/natural_images/car":
#         paths = random.choices(glob(class_path + "/*"), k=700)
#         neg_paths = paths + neg_paths
# 
# print("There are {} car images in the dataset".format(len(car_paths)))
# print("There are {} negative images in the dataset".format(len(neg_paths)))
# 
# example_image = np.asarray(PIL.Image.open(car_paths[0]))
# hog_features,visualized = hog(example_image,orientations=9,pixels_per_cell=(16,16),
#                               cells_per_block=(2,2),
#                               visualize=True,
#                               multichannel=True
#                              )
# 
# fig = plt.figure(figsize=(12,6))
# fig.add_subplot(1,2,1)
# plt.imshow(example_image)
# plt.axis("off")
# fig.add_subplot(1,2,2)
# plt.imshow(visualized,cmap="gray")
# plt.axis("off")
# plt.show()
# 
# pos_images = []
# neg_images = []
# 
# pos_labels = np.ones(len(car_paths))
# neg_labels = np.zeros(len(neg_paths))
# 
# start = time.perf_counter( )
# 
# for car_path in car_paths:
#     img = np.asarray(PIL.Image.open(car_path))
#     # We don't have to use RGB channels to extract features, Grayscale is enough.
#     img = cv2.cvtColor(cv2.resize(img, (96, 64)), cv2.COLOR_RGB2GRAY)
#     img = hog(img, orientations=9, pixels_per_cell=(16, 16),
#               cells_per_block=(2, 2)
#               )
# 
#     pos_images.append(img)
# 
# for neg_path in neg_paths:
#     img = np.asarray(PIL.Image.open(neg_path))
#     img = cv2.cvtColor(cv2.resize(img, (96, 64)), cv2.COLOR_RGB2GRAY)
#     img = hog(img, orientations=9, pixels_per_cell=(16, 16),
#               cells_per_block=(2, 2)
#               )
# 
#     neg_images.append(img)
# 
# x = np.asarray(pos_images + neg_images)
# y = np.asarray(list(pos_labels) + list(neg_labels))
# 
# processTime = round(time.perf_counter( ) - start, 2)
# print("Reading images and extracting features has taken {} seconds".format(processTime))
# 
# print("Shape of image set", x.shape)
# print("Shape of labels", y.shape)
# 
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# 
# # Скользящее окно
# 
# def slideExtract(image, windowSize=(96, 64), channel="RGB", step=12):
#     # Converting to grayscale
#     if channel == "RGB":
#         img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     elif channel == "BGR":
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     elif channel.lower( ) != "grayscale" or channel.lower( ) != "gray":
#         raise Exception("Invalid channel type")
# 
#     # We'll store coords and features in these lists
#     coords = []
#     features = []
# 
#     hIm, wIm = image.shape[:2]
# 
#     # W1 will start from 0 to end of image - window size
#     # W2 will start from window size to end of image
#     # We'll use step (stride) like convolution kernels.
#     for w1, w2 in zip(range(0, wIm - windowSize[0], step), range(windowSize[0], wIm, step)):
# 
#         for h1, h2 in zip(range(0, hIm - windowSize[1], step), range(windowSize[1], hIm, step)):
#             window = img[h1:h2, w1:w2]
#             features_of_window = hog(window, orientations=9, pixels_per_cell=(16, 16),
#                                      cells_per_block=(2, 2)
#                                      )
# 
#             coords.append((w1, w2, h1, h2))
#             features.append(features_of_window)
# 
#     return (coords, np.asarray(features))
# 
# 
# from sklearn.preprocessing import MinMaxScaler
# 
# 
# class Heatmap( ):
# 
#     def __init__(self, original_image):
#         # Mask attribute is the heatmap initialized with zeros
#         self.mask = np.zeros(original_image.shape[:2])
# 
#     # Increase value of region function will add some heat to heatmap
#     def incValOfReg(self, coords):
#         w1, w2, h1, h2 = coords
#         self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] + 30
# 
#     # Decrease value of region function will remove some heat from heatmap
#     # We'll use this function if a region considered negative
#     def decValOfReg(self, coords):
#         w1, w2, h1, h2 = coords
#         self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] - 30
# 
#     def compileHeatmap(self):
#         # As you know,pixel values must be between 0 and 255 (uint8)
#         # Now we'll scale our values between 0 and 255 and convert it to uint8
# 
#         # Scaling between 0 and 1
#         scaler = MinMaxScaler( )
# 
#         self.mask = scaler.fit_transform(self.mask)
# 
#         # Scaling between 0 and 255
#         self.mask = np.asarray(self.mask * 255).astype(np.uint8)
# 
#         # Now we'll threshold our mask, if a value is higher than 170, it will be white else
#         # it will be black
#         self.mask = cv2.inRange(self.mask, 170, 255)
# 
#         return self.mask
# 
# 
# def detect(image):
#     # Extracting features and initalizing heatmap
#     coords, features = slideExtract(image)
#     htmp = Heatmap(image)
# 
#     for i in range(len(features)):
#         # If region is positive then add some heat
#         decision = svc.predict([features[i]])
#         if decision[0] == 1:
#             htmp.incValOfReg(coords[i])
#             # Else remove some heat
#         else:
#             htmp.decValOfReg(coords[i])
# 
#     # Compiling heatmap
#     mask = htmp.compileHeatmap( )
# 
#     cont, _ = cv2.findContours(mask, 1, 2)[:2]
#     for c in cont:
#         # If a contour is small don't consider it
#         if cv2.contourArea(c) < 70 * 70:
#             continue
# 
#         (x, y, w, h) = cv2.boundingRect(c)
#         image = cv2.rectangle(image, (x, y), (x + w, y + h), (255), 2)
# 
#     return image
# 
# 
# detected = detect(np.asarray(PIL.Image.open(путь_к_картинке)))
# plt.imshow(detected)