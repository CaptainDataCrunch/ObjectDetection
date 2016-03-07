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

def save_params(alphas, best_models, best_blocks, best_features, error_rate_list, error_dict):
	# saving parameters to files
	pickle.dump(alphas, open("save_alphas.p", "wb"))
	pickle.dump(best_blocks, open("save_blocks.p", "wb"))
	pickle.dump(best_models, open("save_models.p", "wb"))
	pickle.dump(best_features, open("save_features.p", "wb"))
	pickle.dump(error_rate_list, open("save_error_rate_list.p", "wb"))
	pickle.dump(error_dict, open("save_error_dict.p", "wb"))
	return "Parameters saved!"

def load_params(alpha_file='save_alphas.p', model_file='save_models.p', block_file='save_blocks.p', feature_file='save_features.p'):
	test_alphas = pickle.load(open(alpha_file, "rb"))
	test_blocks = pickle.load(open(block_file, "rb"))
	test_models = pickle.load(open(model_file, "rb"))
	test_features = pickle.load(open(feature_file, "rb"))
	return test_alphas, test_models, test_blocks, test_features
