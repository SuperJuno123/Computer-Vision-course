import cv2

import matplotlib.pyplot as plt

file_name = "cameraman.tif"

img = cv2.imread(file_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

v = img[250, :]

img = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=100)
import numpy as np

img1 = cv2.filter2D(img, ddepth=-1, kernel=np.array([[-1, 0, 1],
                                                     [-2, 0, 2],
                                                     [-1, 0, 1]]))
img2 = cv2.filter2D(img, ddepth=-1, kernel=np.array([[-1, -2, -1],
                                                     [0, 0, 0],
                                                     [1, 2, 1]]))
plt.figure()
plt.imshow(img1, cmap='gray')
plt.show()
#

plt.figure()
plt.imshow(img2, cmap='gray')
plt.show()

G = np.sqrt(img1 ** 2 + img2 ** 2) # Магнитуда
plt.figure()
plt.imshow(G, cmap='gray')
plt.show()


while (True):
    print("Введите порог")
    threshold = int(input())

    GT = G > threshold # True-False matrix

    boundaries = np.ones(shape=GT.shape)
    boundaries[GT == True] = 0

    # boundaries=boundaries*255

    plt.figure()
    plt.imshow(boundaries, cmap='gray')
    plt.show()
