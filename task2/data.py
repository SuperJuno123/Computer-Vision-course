import cv2
import pickle


def extract_data():
    print("Please wait...")
    import tarfile
    tar = tarfile.open("faces.tar.gz", "r:gz")  # https://docs.python.org/3/library/tarfile.html
    tar.extractall()

    import os
    for file in os.listdir(os.getcwd()):
        if file.endswith(".gz") and file != "faces.tar.gz":
            tar_file = tarfile.open(file, "r:gz")
            tar_file.extractall()
            print("{0} extracted succesfully".format(file))
            tar_file.close()
    tar.close()
    print("Complete!")


def read_image(path):
    im_cv2 = cv2.imread(path)
    im_cv2_gray = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2GRAY)
    return im_cv2_gray


def from_pickle():
    with open('data.pickle', 'rb') as f:
        new_data = pickle.load(f)
    return new_data


def to_pickle(data):
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)


def standartization(image):
    """shape (19, 19) -> (24, 24)"""
    return cv2.resize(image, dsize=(24, 24))
