__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

import numpy as np
from sklearn import tree
import time
import sys

# from adaboostFeatures import *
from adaboostFeatures2 import *
from adaboostImages import *

sys.setrecursionlimit(10000)


def calculate_error(prediction, label):
    """Calculates error of classification
    :param prediction: label given from classifier
    :param label: actual label
    :return: 0 if correct, 1 if incorrect
    """
    if prediction == label:
        return 0
    else:
        return 1


def calculate_alpha(error):
    """Calculates weight used to update the distribution
    :param error_counter: sum of incorrect classifications
    :return: weight
    """
    alpha = 0
    if error != 0:
        alpha = (.5) * np.log((1 - error) / error)
    return alpha


def normalization_constant(dists):
    """Calculates constant to normalize distribution so that the integral sums to 1
    :param dists: List of distribution values
    :return: constant value
    """
    normalization = sum(dists)
    return normalization


def get_feature_values(gray_imgs, features):
    """ Obtains the integral images values for a given set of features and images
    """
    integral_images_dict = dict()
    start1 = time.time()
    for gray_img in gray_imgs:
        blocks = partition_image(gray_img)
        for i, feature in enumerate(features):
            for j, block in enumerate(blocks):
                key_img = (i, j)
                diff = feature(gray_img, block)
                if key_img in integral_images_dict:
                    integral_images_dict[key_img].append(diff)
                else:
                    integral_images_dict[key_img] = [diff]
    end1 = time.time()
    print "time of triple loop is:", ((end1 - start1) / 60), "min"
    return integral_images_dict

def features_size():
    size_dict = dict()
    size_dict[feat_two_rectangles] = (24, 28)
    size_dict[feat_three_rectangles_horizontal] = (24, 62)
    size_dict[feat_three_rectangles_vertical] = (128, 16)
    size_dict[feat_four_rectangles] = (28, 28)

    return size_dict


def get_feature_values2(gray_imgs, feature, feat_height, feat_width, width=150, height=120, increment=5,
                        feat_dict=None):
    if feat_dict is None:
        feat_dict = dict()

    x0 = 0
    y0 = 0
    x1 = feat_width
    y1 = feat_height

    while x1 < width:
        while y1 < height:
            coords = (x0, y0, x1, y1)
            #print coords
            feat_dict[(feature, coords)] = [feature(x, coords) for x in gray_imgs]
            y0 += increment
            y1 += increment
        x0 += increment
        x1 += increment
        y0 = 0
        y1 = feat_height

    return feat_dict


def loop_features(gray_imgs, features, size_dict):
    feat_dict = None
    for feat in features:
        print feat
        feat_width, feat_height = size_dict[feat]
        feat_dict = get_feature_values2(gray_imgs, feat, feat_height, feat_width, feat_dict = feat_dict)

    return feat_dict

def create_error_dict(integral_images_dict, labels, distribution):
    error_dict = dict()
    for k, v in integral_images_dict.items():
        X = v
        X_list = [[item] for item in X]
        clf1 = tree.DecisionTreeClassifier(max_depth=1)
        clf = clf1.fit(X_list, labels)
        predictions = clf.predict(X_list)
        # print predictions.tolist()
        # print predictions.tolist()[0]
        # print type(predictions), type(predictions.tolist()), type(predictions.tolist()[0])
        incorrectly_classified = [x[0] != x[1] for x in zip(predictions.tolist(), labels)]
        error_rate = sum([x[0] * x[1] for x in zip(distribution, incorrectly_classified)])
        error_dict[(clf, k[0], k[1])] = (error_rate, predictions)
        # print v
        #print "error rate of key", k, "is", error_rate
    return error_dict


def weak_learner(gray_imgs, integral_images_dict, features, labels, distribution, error_dict):
    """Creates decision trees for each feature and each block and calculates training error. Selects the best model for a specific feature and block based on lowest training error.
    :param gray_imgs: list where each element is an array of pixels
    :param integral_images_dict:
    :param features: list of function names to caculate features
    :param labels: list of 1's or -1's
    :param distribution: list of weights for each image
    :return: tuple of (best_model, best_block, best_feature, lowest_error_rate, correctly_classified)
    """
    best_feature = []
    correctly_classified = []
    lowest_error_rate = 1.0
    best_model = []
    best_coords = []

    start2 = time.time()

    for k, v in error_dict.items():
        error_rate = v[0]
        predictions = v[1]
        if error_rate < lowest_error_rate:
            best_feature = k[1]
            best_coords = k[2]
            lowest_error_rate = error_rate
            best_model = k[0]
            correctly_classified = [x[0] == x[1] for x in zip(predictions, labels)]
    end2 = time.time()
    print "time of classification loop is:", ((end2 - start2) / 60), "min"
    return (best_model, best_coords, best_feature, lowest_error_rate, correctly_classified)


def adaboost_train(pos_filepath, neg_filepath, T=3):
    """Performs adaboost on training set
    :param pos_filepath: directory of positive files
    :param neg_filepath: directory of negative files
    :return: tuple of (alphas, best_models, best_coords, best_features) where alphas are the weights of the models
    """
    images = get_gray_imgs(pos_filepath, neg_filepath)

    gray_imgs = [x[0] for x in images]
    labels = [x[1] for x in images]

    n = len(gray_imgs)

    dists = [1.0 / n for i in range(n)]

    alphas = list()

    features = [feat_two_rectangles, feat_three_rectangles_horizontal, feat_three_rectangles_vertical,
                feat_four_rectangles]
    #features = [feat_four_rectangles]
    size_dict = features_size()
    integral_images_dict = loop_features(gray_imgs, features, size_dict)

    best_models = list()
    best_features = list()
    best_coords = list()
    error_rate_list = list()
    for t in range(T):
        print t
        error_dict = create_error_dict(integral_images_dict, labels, dists)
        best_model, best_coord, best_feature, lowest_error_rate, correctly_classified = weak_learner(gray_imgs,
                                                                                                     integral_images_dict,
                                                                                                     features, labels,
                                                                                                     dists, error_dict)
        error_rate_list.append(lowest_error_rate)
        # print best_model, best_block, best_feature, lowest_error_rate, correctly_classified
        print lowest_error_rate
        best_models.append(best_model)
        best_features.append(best_feature)
        best_coords.append(best_coord)
        alpha = calculate_alpha(lowest_error_rate)
        alphas.append(alpha)
        for i in range(n):
            if correctly_classified[i] == True:
                dists[i] = dists[i] * np.exp(
                        -alpha)  # update distributions for each image based on if they are correctly classified
            else:
                dists[i] = dists[i] * np.exp(alpha)
        normalization = normalization_constant(dists)
        dists = [x / normalization for x in dists]
    # print "distributution", dists
    return alphas, best_models, best_coords, best_features, error_rate_list, error_dict
