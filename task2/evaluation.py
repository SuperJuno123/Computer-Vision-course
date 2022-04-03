import numpy as np

mode = 'mute'


def feature_two_vert_lines(int_img):
    counter = 0
    characteristics = []
    for i in range(24):
        for j in range(24):
            for height in range(1, 24):
                if i + height < 24:
                    for width in range(1, 24):
                        if 2 * width + j < 24:
                            s1 = eval_area(int_img, y0=i, x0=j, width=width, height=height)
                            s2 = eval_area(int_img, y0=i, x0=j + width, width=width, height=height)
                            counter += 1
                            characteristics.append(s1 - s2)
    # Согласно статье, должо быть 43200 features для данной категории
    if not mode == 'mute':
        print("Всего было вычислено {0} features of category 'a' (two adjacent vertical area)".format(counter))
    return characteristics


def feature_three_vert_lines(int_img):
    counter = 0
    characteristics = []
    for i in range(24):
        for j in range(24):
            for height in range(1, 24):
                if i + height < 24:
                    for width in range(1, 24):
                        if 3 * width + j < 24:
                            s1 = eval_area(int_img, y0=i, x0=j, width=width, height=height)
                            s2 = eval_area(int_img, y0=i, x0=j + width, width=width, height=height)
                            s3 = eval_area(int_img, y0=i, x0=j + 2 * width, width=width, height=height)
                            counter += 1
                            characteristics.append(s1 - s2 + s3)
    # Согласно статье, должо быть 27600 features для данной категории
    if not mode == 'mute':
        print("Всего было вычислено {0} features of category 'b' (three adjacent vertical area)".format(counter))
    return characteristics


def feature_two_horiz_lines(int_img):
    counter = 0
    characteristics = []
    for i in range(24):
        for j in range(24):
            for height in range(1, 24):
                if i + 2 * height < 24:
                    for width in range(1, 24):
                        if width + j < 24:
                            s1 = eval_area(int_img, y0=i, x0=j, width=width, height=height)
                            s2 = eval_area(int_img, y0=i + height, x0=j, width=width, height=height)
                            counter += 1
                            characteristics.append(int(s1) - int(s2))
    # Согласно статье, должо быть 43200 features для данной категории
    if not mode == 'mute':
        print("Всего было вычислено {0} features of category 'c' (two adjacent horizontal area)".format(counter))
    return characteristics


def feature_three_horiz_lines(int_img):
    counter = 0
    characteristics = []
    for i in range(24):
        for j in range(24):
            for height in range(1, 24):
                if i + 3 * height < 24:
                    for width in range(1, 24):
                        if width + j < 24:
                            s1 = eval_area(int_img, y0=i, x0=j, width=width, height=height)
                            s2 = eval_area(int_img, y0=i + height, x0=j, width=width, height=height)
                            s3 = eval_area(int_img, y0=i + 2 * height, x0=j, height=height, width=width)
                            counter += 1
                            characteristics.append(s1 - s2 + s3)
    # Согласно статье, должо быть 23700 features для данной категории
    if not mode == 'mute':
        print("Всего было вычислено {0} features of category 'd' (three adjacent horizontal area)".format(counter))
    return characteristics


def feature_four_square(int_img):
    counter = 0
    characteristics = []
    for i in range(24):
        for j in range(24):
            for height in range(1, 24):
                if i + 2 * height < 24:
                    for width in range(1, 24):
                        if 2 * width + j < 24:
                            s1 = eval_area(int_img, y0=i, x0=j, width=width, height=height)
                            s2 = eval_area(int_img, y0=i + height, x0=j, width=width, height=height)
                            s3 = eval_area(int_img, y0=i, x0=j + width, width=width, height=height)
                            s4 = eval_area(int_img, y0=i + height, x0=j + width, width=width, height=height)
                            counter += 1
                            characteristics.append(s4 + s1 - s3 - s3)
    # Согласно статье, должо быть 20736 features для данной категории
    if not mode == 'mute':
        print("Всего было вычислено {0} features of category 'e' (four adjacent square)".format(counter))
    return characteristics


LIST_OF_FEATURE = [feature_two_vert_lines,  # a
                   feature_three_vert_lines,  # b
                   feature_two_horiz_lines,  # c
                   feature_three_horiz_lines,  # d
                   feature_four_square]  # e


def eval_training_characterictics_of_all_pics():
    import os
    matrix = []
    counter = 0
    for file_name in os.listdir(os.getcwd() + '\\train\\face'):
        # if counter > 0:
        #     break
        matrix.append(eval_characteristic_vector_of_one_pic(os.getcwd() + '\\train\\face\\' + file_name))
        print(counter)
        counter += 1
    return matrix


def eval_characteristic_vector_of_one_pic(path):
    import data
    image = data.standartization(data.read_image(path))

    characteristics = []

    integral_img = create_integral_image(image)

    for compute_feature in LIST_OF_FEATURE:
        characteristics.extend(compute_feature(integral_img))

    return characteristics


def create_integral_image(image):
    return np.cumsum(np.cumsum(image, axis=0), axis=1)


def eval_area(int_img, x0, y0, width, height):
    A = int_img[x0, y0]
    B = int_img[x0 + width, y0]
    C = int_img[x0, y0 + height]
    D = int_img[x0 + width, y0 + height]
    return int(D - B - C + A)
