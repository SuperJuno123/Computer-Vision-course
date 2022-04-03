import cv2
import numpy as np
import matplotlib.pyplot as plt

# region parameters

filename = "lena_color_512.tif"
sigmaForGaussianBlur = 100

# endregion

img = cv2.imread(filename)

# 0. Перевести изображение в полутоновое (серое)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

# 1. Размыть изображение фильтром Гаусса с небольшой дисперсией
imgGaussian = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=sigmaForGaussianBlur, sigmaY=sigmaForGaussianBlur)

# 2. Вычислить градиент изображения с помощью фильтра Собеля
G_OX = cv2.filter2D(imgGaussian, ddepth=-1, kernel=np.array([[-1, 0, 1],
                                                             [-2, 0, 2],
                                                             [-1, 0, 1]]))
G_OY = cv2.filter2D(imgGaussian, ddepth=-1, kernel=np.array([[-1, -2, -1],
                                                             [0, 0, 0],
                                                             [1, 2, 1]]))

# 3. Вычислить магнитуду и направление градиента
G = np.sqrt(G_OX ** 2 + G_OY ** 2)
grad_vect = np.arctan2(G_OY, G_OX)

# 4. Округлить направление градиента до 8 направлений (0, 45, 90, 135, 180, 225, 270, 315)
import evaluation as eval
grad_vect_rounded = eval.rounding_angles(grad_vect)

# 5. Применить процедуру NMS (подавление немаксимумов) вдоль направления градиентов
with_NMS = eval.NMS_grad(imgGaussian, grad_vect_rounded)

# 6. Применить процедуру гистерезиса на основе 2-х границ.
upper_threshold = 200
lower_threshold = 100

after_hyster = eval.hysteresis(with_NMS, upper_threshold, lower_threshold)

# Демонстрация промежуточных результатов программы
fig, axes = plt.subplots(1, 2)


axes[0].imshow(img, cmap='gray')
axes[0].set_title("Исходная картинка")

axes[1].imshow(imgGaussian, cmap='gray')
axes[1].set_title("После применения фильтра Гаусса")

plt.show()

fig, axes = plt.subplots(1, 3)

plt.title("Результат применения оператора Собеля в горизонтальном и вертикальном направлениях, усредненное значение проекций")

axes[0].imshow(G_OX,cmap='gray')
axes[1].imshow(G_OY,cmap='gray')
axes[2].imshow(G, cmap='gray')

plt.show()

fig, axes = plt.subplots(1, 3)

axes[0].imshow(grad_vect, cmap='gray')
axes[0].set_title("Градиент")
axes[1].imshow(with_NMS, cmap='gray')
axes[1].set_title("NMS")
axes[2].imshow(after_hyster, cmap='gray')
axes[2].set_title("hystero")

plt.show()