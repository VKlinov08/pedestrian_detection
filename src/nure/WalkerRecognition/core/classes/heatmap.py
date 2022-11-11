from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2


class Heatmap():

    def __init__(self, original_image):
        # Mask attribute is the heatmap initialized with zeros
        self.mask = np.zeros(original_image.shape[:2])

    # Increase value of region function will add some heat to heatmap
    def incValOfReg(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] + 50

    # Decrease value of region function will remove some heat from heatmap
    # We'll use this function if a region considered negative
    def decValOfReg(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] - 30

    def compileHeatmap(self):
        # As you know,pixel values must be between 0 and 255 (uint8)
        # Now we'll scale our values between 0 and 255 and convert it to uint8

        # Scaling between 0 and 1
        scaler = MinMaxScaler( )

        self.mask = scaler.fit_transform(self.mask)

        # Scaling between 0 and 255
        self.mask = np.asarray(self.mask * 255).astype(np.uint8)

        # Now we'll threshold our mask, if a value is higher than 170, it will be white else
        # it will be black
        self.mask = cv2.inRange(self.mask, 170, 255)

        return self.mask









