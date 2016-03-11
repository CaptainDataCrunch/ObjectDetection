__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

from adaboostImages import *
import numpy as np
from ggplot import *
import pandas as pd

filepath = "/Users/vincentpham/CaptainDataCrunch-/main/final_model/"
pos_trainpath = "/Users/vincentpham/Desktop/tester/final_pos_train/"
neg_trainpath = "/Users/vincentpham/Desktop/tester/final_neg_train/"

pos_testpath = "/Users/vincentpham/Desktop/tester/final_pos_test/"
neg_testpath = "/Users/vincentpham/Desktop/tester/final_neg_test/"

def calculate_products(gray_img, alphas, best_models, best_coords, best_features, T = None):
    """Calculates the products of alphas * intergral image
    :param gray_img: array of pixel values
    :param alphas: list where the elements are weights wegihts of the models
    :param best_models: list of models selected by adaboost
    :param best_coords: list of blocks selected by adaboost
    :param best_features: list of features selected by adaboost
    :param T: number of iteration to look at
    :return: List of products
    """
    if T is None:
        T = len(alphas)

    haar = [best_features[i](gray_img, best_coords[i]) for i in range(T)]
    haar = [[x] for x in haar]
    predictions = [best_models[i].predict(haar[i]) for i in range(T)]

    products = [x[0] * x[1] for x in zip(predictions, alphas)]
    return products


def get_training_accuracy(pos_testpath, neg_testpath, trained_model, T = None):
    """ test files in test directory
    :param pos_testpath: directory of positive files
    :param neg_testpath: directory of negative files
    :param trained_model: trained model of (alphas, models, coords, features)
    :param T: number of iteration to look at
    :return:
    """
    images = get_gray_imgs(pos_testpath, neg_testpath)
    if T == None:
        T = len(trained_model[0])

    #accuracy_list = list()
    # for i in range(T):

    positive_counter = [0]*T
    negative_counter = [0]*T
    pos_acc = [0]*T
    neg_acc = [0]*T

    print_count = 0
    for gray_img, label in images:
        print print_count
        print_count += 1

        products = calculate_products(gray_img, trained_model[0], trained_model[1], trained_model[2],
                                                trained_model[3])
        for i in range(T):
            if label == 1:
                positive_counter[i] += 1.0
            elif label == -1:
                negative_counter[i] += 1.0

            prediction = np.sign(sum(products[:(i+1)]))
            if prediction == label and label == 1:
                pos_acc[i] += 1.0
            if prediction == label and label == -1:
                neg_acc[i] += 1.0

        #accuracy_list.append([i, pos_acc, neg_acc, positive_counter, negative_counter])
    d = {"T":pd.Series([i for i in range(T)]), "pos_acc_count":pd.Series(positive_counter),
         "neg_acc_count":pd.Series(negative_counter),
         "pos_count":pd.Series(pos_acc),"neg_count":pd.Series(neg_acc)}
    df = pd.DataFrame(d)
    return df

alphas = load_param(filepath, "save_alphas.p")
coords = load_param(filepath, "save_blocks.p")
error_rate_list = load_param(filepath, "save_error_rate_list.p")
features = load_param(filepath, "save_features.p")
models = load_param(filepath, "save_models.p")
print "alphas", alphas
print
print "coords", coords
print
print "errors", error_rate_list
print
print "features", features
#print "models", models
n = len(alphas)


d = {"T":pd.Series([i for i in range(n)]), "error":pd.Series(error_rate_list),
     "alpha":pd.Series(alphas)}
df = pd.DataFrame(d)

#plots distribution weighted error at each iteration
print ggplot(df,aes(x='T', y='error')) + geom_point(color = "blue") \
      + geom_line(color = "lightblue") + ggtitle("Distribution Weighted Error") + theme_matplotlib()

#plots alphas value at each iteration
print ggplot(df, aes(x = "T", y = "alpha")) + geom_point(color = "blue") + ggtitle("Alphas at Iteration T") \
          + geom_line(color = "lightblue")  + theme_matplotlib()

#plots the training error
prediction_df = get_training_accuracy(pos_trainpath, neg_trainpath, (alphas, models, coords, features), T = None)

prediction_df["pos_error"] = 1- prediction_df["pos_count"]/ prediction_df["pos_acc_count"]
prediction_df["neg_error"] = 1- prediction_df["neg_count"]/ prediction_df["neg_acc_count"]

#blue["positive_accuracy"] = blue["pos_count"]/ blue["pos_acc_count"]

pos_df = prediction_df[["T","pos_error"]]
pos_df["labels"] = "+"
pos_df.rename(columns = {"pos_error":"error"}, inplace = True)

neg_df = prediction_df[["T","neg_error"]]
neg_df["labels"] = "-"
neg_df.rename(columns = {"neg_error":"error"}, inplace = True)


df_train = pos_df.append(neg_df)

print ggplot(df_train,aes(x='T', y='error', color = "labels")) +geom_point() + geom_line() + ggtitle("training error per iteration")

#plots the testing error
prediction_df_test = get_training_accuracy(pos_testpath, neg_testpath, (alphas, models, coords, features), T = None)

prediction_df_test["pos_error"] = 1- prediction_df_test["pos_count"]/ prediction_df_test["pos_acc_count"]
prediction_df_test["neg_error"] = 1- prediction_df_test["neg_count"]/ prediction_df_test["neg_acc_count"]

#blue["positive_accuracy"] = blue["pos_count"]/ blue["pos_acc_count"]

pos_df = prediction_df_test[["T","pos_error"]]
pos_df["labels"] = "+"
pos_df.rename(columns = {"pos_error":"error"}, inplace = True)

neg_df = prediction_df_test[["T","neg_error"]]
neg_df["labels"] = "-"
neg_df.rename(columns = {"neg_error":"error"}, inplace = True)


df_test = pos_df.append(neg_df)

print ggplot(df_test,aes(x='T', y='error', color = "labels")) +geom_point() + geom_line() + ggtitle("testing error per iteration")
