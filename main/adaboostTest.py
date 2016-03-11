__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

import numpy as np
import sys

from adaboostImages import *

sys.setrecursionlimit(10000)

def calculate_final_hypothesis(gray_img, alphas, best_models, best_coords, best_features, scale=1.0):
    """Calculates the classification for a test image
    :param gray_img: array of pixel values
    :param alphas: list where the elements are weights wegihts of the models
    :param best_models: list of models selected by adaboost
    :param best_coords: list of blocks selected by adaboost
    :param best_features: list of features selected by adaboost
    :return: classification for test image
    """
    T = len(alphas)

    haar = list()
    for i in range(T):
        width1 = 52 # original widths for vertical feature for scaling
        width2 = 24
        height1 = 24 # original heights for horizontal feature for scaling
        height2 = 14
        if 'vertical' in best_features[i].__str__():
            width1 = int(width1*scale)
            width2 = int(width2*scale)
            haar.append(best_features[i](gray_img, best_coords[i], width1, width2))
        elif 'horizontal' in best_features[i].__str__():
            height1 = int(height1*scale)
            height2 = int(height2*scale)
            haar.append(best_features[i](gray_img, best_coords[i], height1, height2))
        else:
            haar.append(best_features[i](gray_img, best_coords[i]))

    # haar = [best_features[i](gray_img, best_coords[i]) for i in range(T)]
    haar = [[x] for x in haar]
    predictions = [best_models[i].predict(haar[i]) for i in range(T)]

    products = [x[0] * x[1] for x in zip(predictions, alphas)]

    classification = np.sign(sum(products))
    return classification

def test_training_images(pos_testpath, neg_testpath, trained_model):
    """ test files in test directory
    :param pos_testpath: directory containing positive test files
    :param neg_testpath: directory containing negative test files
    :param trained_model: training model
    :return:
    """
    images = get_gray_imgs(pos_testpath, neg_testpath)

    positive_counter = 0
    negative_counter = 0
    pos_acc = 0
    neg_acc = 0

    for gray_img, label in images:
        if label == 1:
            positive_counter += 1.0
        elif label == -1:
            negative_counter += 1.0

        prediction = calculate_final_hypothesis(gray_img, trained_model[0], trained_model[1], trained_model[2],
                                                trained_model[3])

        if prediction == label and label == 1:
            pos_acc += 1.0
        if prediction == label and label == -1:
            neg_acc += 1.0

    print "positive accuracy", pos_acc / positive_counter
    print "negative accuracy", neg_acc / negative_counter
    print "overall accuracy", (pos_acc + neg_acc) / (positive_counter + negative_counter)
    return
