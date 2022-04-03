Задание 3. Алгоритм Canny.


Реализовать алгоритм Canny для детектирования границ на изображении.

0. Перевести изображение в полутоновое (серое)
1. Размыть изображение фильтром Гаусса с небольшой дисперсией
2. Вычислить градиент изображения с помощью фильтра Собеля
3. Вычислить магнитуду и направление градиента
4. Округлить направление градиента до 8 направлений (0, 45, 90, 135, 180, 225, 270, 315)
5. Применить процедуру NMS (подавление немаксимумов) вдоль направления градиентов
6. Применить процедуру гистерезиса на основе 2-х границ. Пиксели, для которых магнитуда градиента больше верхнего порога объявляются границами. Пиксели, для которых магнитуда меньше нижнего порога, объявляются фоном. Пиксели, для которых магнитуда градиента лежит между порогами, проверяются на соседство с пикселем границы. 


Для подсчета градиента использовать фильтры Собеля 3x3. Подавление немаксимумов произвести вдоль направления градиента на расстоянии 1 пиксель. На этапе гистерезиса использовать окрестность в 1 пиксель для проверки наличия пикселей со значением градиента выше верхнего порога. В качестве порогов взять значения 100 и 200 соответвенно. Отобразить результаты каждого из этапов алгоритма.