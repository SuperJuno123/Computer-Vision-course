import numpy as np


def rounding_angles(G):
    """G принимает значения от -3.14 до +3.14"""
    possible_angle = [-np.pi, -3 * np.pi / 4, -np.pi / 2, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    for angle in possible_angle:
        a = G >= angle - np.pi
        b = G >= angle + np.pi
        c = a & b
        G[c] = angle
        # G[G>=angle-np.pi & G<=angle+np.pi] = angle
    #  Предыдущая строка не срабатывает
    return G


def NMS_naive(img, n):
    """Naive implementation"""
    newMatrix = np.zeros(img.shape)

    for i in range(n, img.shape[0] - n):
        for j in range(n, img.shape[1] - n):
            subwindow = subwindow(img, (i, j), n)
            index = np.argmax(subwindow)
            local_indexes = divmod(index, subwindow.shape[0])  # (частное, остаток)
            newMatrix[local_indexes[0] + i, local_indexes[1] + j] = np.max(subwindow)
    return newMatrix


def NMS_grad(img, grad, n=1):
    newMatrix = np.zeros(img.shape)
    for i in range(n, img.shape[0]):
        for j in range(n, img.shape[0]):
            # TODO: оформить в виде списка кортежей
            dx = dy = 0
            if grad[i, j] == 0:
                dx = 1
                dy = 0
            elif grad[i, j] == np.pi / 4:
                dx = 1
                dy = -1
            elif grad[i, j] == np.pi / 2:
                dx = 0
                dy = -1
            elif grad[i, j] == 3 * np.pi / 4:
                dx = -1
                dy = -1
            elif grad[i, j] == np.pi:
                dx = -1
                dy = 0
            elif grad[i, j] == -np.pi / 4:
                dx = 1
                dy = 1
            elif grad[i, j] == -np.pi / 2:
                dx = 0
                dy = 1
            elif grad[i, j] == -3 * np.pi / 4:
                dx = -1
                dy = 1

            if img[i, j] < img[i + dy, j + dx]:
                newMatrix[i + dy, j + dx] = img[i + dy, j + dx]
            else:
                newMatrix[i, j] = img[i, j]

    return newMatrix


def subwindow(matrix, center, bias):
    return matrix[center[0] - bias:center[0] + bias, center[1] - bias: center[1] + bias]


# Кэнни ввёл понятие подавления немаксимумов (англ. Non-Maximum Suppression), которое означает, что пикселями границ
# объявляются пиксели, в которых достигается локальный максимум градиента в направлении вектора градиента.

def hysteresis(img, up_t, low_t):
    """Пиксели, для которых магнитуда градиента больше верхнего порога объявляются границами (True).
    Пиксели, для которых магнитуда меньше нижнего порога, объявляются фоном (False). Пиксели,
     для которых магнитуда градиента лежит между порогами, проверяются на соседство с пикселем границы."""
    truefalseMatrix = np.zeros(img.shape)
    truefalseMatrix[img > up_t] = True
    truefalseMatrix[img < low_t] = False
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j]
            if pixel > up_t:
                img[i, j] = True
            elif pixel < low_t:
                img[i, j] = False
            elif isNeigboorIsEdge(img, (i, j)):
                img[i, j] = True
            else:
                img[i, j] = False

    return truefalseMatrix


def isNeigboorIsEdge(img, pixel_coord):
    neigbors = subwindow(img, pixel_coord, 1)
    return (neigbors == True).any()
