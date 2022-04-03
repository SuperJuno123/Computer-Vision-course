import numpy as np
import cv2
import matplotlib.pyplot as plt


# region evaluations.py

def rgb_to_gray(image):
    """
    Получение черно-белого изображения путем усреднения каналов
    :param image: ndarray, (hight, width, 3)
    :return: ndarray, (hight, width)
    """
    return (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3


def add_noise(image, sigma=0.1):
    """
    По умолчанию - сигма, соответствующая стандартному распределению. Добавление шума из случайного распределения
    :param image: ndarray(height, width), изображение со значениями от 0 до 255
    :param sigma: стандартное отклонение
    :return: ndarray(height, width) со значениями от 0 до 255
    """
    new_im = image + np.random.normal(0, sigma, size=image.shape) * 255
    new_im[new_im < 0] = 0
    new_im[new_im > 255] = 255
    return new_im


def calculate_color_histogram(image):
    """
    :param image: ndarray (height, width)
    :return: ndarray(256,)
    """
    brightnesses = np.zeros(256)  # 0...255
    for pixel in image.flatten():
        brightnesses[int(pixel)] += 1
    return brightnesses


def Gaussian_filter(image, window_size, sigma):
    G = create_Gaussian_matrix(window_size, sigma)
    bias = int(window_size / 2)

    # image = prepare_image(image, bias)

    height, width = image.shape

    for y in range(bias, height - bias):
        for x in range(bias, width - bias):
            s = substring_matrix(image, x, y, window_size)
            image[y, x] = convolution(s, G)

    return image


def convolution(matrix1, matrix2):
    return np.sum((matrix1 * matrix2).flatten())


def substring_matrix(big_matrix, center_coord_x, center_coord_y, size_of_new_matrix):
    bias = int(size_of_new_matrix / 2)
    return big_matrix[center_coord_y - bias: size_of_new_matrix + center_coord_y - bias,
           center_coord_x - bias:size_of_new_matrix + center_coord_x - bias]


def create_Gaussian_matrix(size, sigma):
    G = np.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            G[j, i] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((i - 1) ** 2 + (j - 1) ** 2) / (2 * sigma ** 2))
    return G / np.sum(G.flatten())


def prepare_image(image, bias):
    # TODO: заставить работать
    # Для удобства явно выделю строки/столбцы, которые буду копировать для увеличения матрицы
    left = image[:, 0].reshape(-1, 1)
    right = image[:, -1].reshape(-1, 1)
    up = image[0,:]
    down = image[-1,:]
    print(left.shape, right.shape, up.shape, down.shape)

    for i in range(bias):
        image = np.hstack((left, image))
        image = np.hstack((image, right))
        image = np.vstack((up, image))
        image = np.vstack((image, down))

    return image


# endregion



# region visualisation.py

def draw_histogram(image, save=False):
    plt.figure()
    plt.title("Гистограмма зашумленного изображения")
    plt.bar([i for i in range(256)], calculate_color_histogram(image), width=1)
    if save:
        plt.savefig("Гистограмма.png")
    plt.show()

# endregion



# region main.py

path_to_image = "lena_color_512.tif"


im_cv2_rgb = cv2.imread(path_to_image)[:, :, ::-1]
im_cv2_rgb = np.float32(im_cv2_rgb)
im_cv2_rgb_gray = rgb_to_gray(im_cv2_rgb)

plt.figure()
plt.title("Исходное изображение")
plt.imshow(np.int32(im_cv2_rgb), cmap='gray')
plt.show()

plt.figure()
plt.title("Перевод изображения в серое")
plt.imshow(im_cv2_rgb_gray, cmap='gray')
# по умолчнию в матблот либе пытаются применить цветуную палитру
# TODO: решить проблему с тем, что серая картинка не записывается в файл, а остальные записываются
cv2.imwrite('Gray ' + path_to_image, im_cv2_rgb_gray)
plt.show()

plt.figure()
sigma = 0.05
img_with_noise = add_noise(im_cv2_rgb_gray, sigma)
plt.title("Добавление шума на картинку с σ = {0}".format(sigma))
plt.imshow(img_with_noise, cmap='gray')
plt.show()
cv2.imwrite('Noise ' + path_to_image, img_with_noise)

draw_histogram(img_with_noise, save=True)

# plt.title("Для сравнения: результат стандартной функции hist()")
# plt.hist(img_with_noise.flatten(), bins=255)
# plt.show()

plt.figure()
window_size = 5     # Оптимально - 5 на Лене 512
sigma = 9           # Оптимально - 8-10 на Лене 512
plt.title("Размытие изображения ядром Гаусса с σ = {0} и размером окна {1}".format(sigma, window_size))
corrected_img = Gaussian_filter(img_with_noise, window_size, sigma)
plt.imshow(corrected_img, cmap='gray')
cv2.imwrite('Gauss ' + path_to_image, img_with_noise)
plt.show()

# endregion
