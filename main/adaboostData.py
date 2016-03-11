__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

import glob
import pickle


def load_data(filepath, label_value):
    """Loads in image and label
    :param filepath: path of the directory containing all files
    :param label_value: classification label
    :return: list of tuples(ndarray, label)
    """
    files = glob.glob(filepath + '*.jpg')
    labels = [label_value for i in range(len(files))]
    return zip(files, labels)


def save_params(alphas, best_models, best_coords, best_features, error_rate_list, error_dict):
    """Saves parameters to pickle files
    :param alphas: list of weights
    :param best_models: list of models that are the best classifiers 
    :param best_coords: list of coordinates that were best classified
    :param best_features: list of features that classified best
    :param error_rate_list: list of error rates
    :param error_dict: dict where key = model, feature, coord and value = (error, prediction)
    """  # saving parameters to files
    pickle.dump(alphas, open("save_alphas.p", "wb"))
    pickle.dump(best_coords, open("save_coords.p", "wb"))
    pickle.dump(best_models, open("save_models.p", "wb"))
    pickle.dump(best_features, open("save_features.p", "wb"))
    pickle.dump(error_rate_list, open("save_error_rate_list.p", "wb"))
    pickle.dump(error_dict, open("save_error_dict.p", "wb"))
    return "Parameters saved!"


def load_params(alpha_file='save_alphas.p', model_file='save_models.p', coords_file='save_coords.p',
                feature_file='save_features.p'):
    """Loads the pickle files containing the parameters
    :param alpha_file: pickle file containing alpha values
    :param model_file: pickle file containing best models
    :param coords_file: pickle file containing best coordinates
    :param feature_file: pickle file containing best features
    :return: tuple(alphas, models, coords, features)
    """
    test_alphas = pickle.load(open(alpha_file, "rb"))
    test_coords = pickle.load(open(coords_file, "rb"))
    test_models = pickle.load(open(model_file, "rb"))
    test_features = pickle.load(open(feature_file, "rb"))
    return test_alphas, test_models, test_coords, test_features


def load_param(path, file):
    """Load a pickle file
    :param path: directory path of pickle file
    :param file: name of pickle file
    :return: parameter from file
    """
    param = pickle.load(open(path + file, "rb"))
    return param
