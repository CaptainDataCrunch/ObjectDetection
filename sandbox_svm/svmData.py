import glob
import cv2


def load_data(filepath, label_value):
    """Loads in image and label
    :param filepath: path of the directory containing all files
    :param label_value: classification label
    :return: list of tuples(ndarray, label)
    """


    files = glob.glob(filepath + '*.jpg')
    images = list()
    for f in files:
        img = cv2.imread(f)
        images.append(img)
    labels = [label_value for i in range(len(files))]
    return (images, labels)


def load_and_label_data(pos_filepath, neg_filepath):
    """Labels and converts images to gray scale
    :param pos_filepath: path of directory containing all pos images
    :param neg_filepath: path of directory containing all neg images
    :return: list of tuples(gray image, label)
    """
    pos_images = load_data(pos_filepath, 1)
    neg_images = load_data(neg_filepath, -1)
    images = pos_images[0] + neg_images[0]
    labels = pos_images[1] + neg_images[1]
    return images, labels
