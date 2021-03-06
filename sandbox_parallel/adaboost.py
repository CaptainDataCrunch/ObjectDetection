__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

import cv2
import numpy as np
import glob
from sklearn import tree
import time
import sys
import pickle

sys.setrecursionlimit(10000)
imagefile = 'image_0033.jpg'
imagefile2 = 'image_0467.jpg'
imagefile3 = 'image_0294.jpg'
imagefile4 = 'image_0078.jpg'
testimagefile = 'carimagetest.jpg'

# make sure full path
neg_filepath = '/Users/htom/Desktop/MSAN_Spring_2016/AdvML/Project/good_dataset/cars_brad_bg/subset/'
pos_filepath = '/Users/htom/Desktop/MSAN_Spring_2016/AdvML/Project/good_dataset/cars_brad/subset/'

neg_testpath = '/Users/htom/Desktop/MSAN_Spring_2016/AdvML/Project/negative_test/'
pos_testpath = '/Users/htom/Desktop/MSAN_Spring_2016/AdvML/Project/positive_test/'

# Stop deprecated warnings from showing
import warnings

def fxn():
	warnings.warn("deprecated", DeprecationWarning)


def load_data(filepath, label_value):
	"""Loads in image and label
	:param filepath: path of the directory containing all files
	:param label_value: classification label
	:return: list of tuples(ndarray, label)
	"""
	files = glob.glob(filepath + '*.jpg')
	labels = [label_value for i in range(len(files))]
	return zip(files, labels)

def read_image(imagefile):
	"""Change image to grayscale
	:param imagefile: image
	:return: image in grayscale
	"""
	# read in image
	img = cv2.imread(imagefile)

	# change image to grayscale
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	return gray_img
  
def s(gray_img,x,y):
	"""Cumulative row sum to calculate integral image
	:param gray_img: image in gray scale
	:param x: x coordinate
	:param y: y coordinate
	:return: row sum of pixel intensities
	"""
	sums = 0
	if y == -1:
		return sums
	else:
		sums += gray_img[x][y] + s(gray_img,x,y-1)
	return sums

def ii(gray_img,x,y):
	"""Cumulative column sum to calcualte integral image
	:param gray_img: image in gray scale
	:param x: x coordinate
	:param y: y coordinate
	:return: column sum of pixel intensities
	"""
	sums = 0
	if x == -1:
		return sums
	else:
		sums += ii(gray_img,x-1,y) + s(gray_img,x,y)
	return sums
  
def integralImage(gray_img, locations):
	"""Calculates integral image to compute rectangle features
	:param gray_img: image in gray scale
	:param x0, y0, x1, y1: coordinates describing the rectangle
	:return: sum of all the pixels above and to the left of (x1, y1)
	"""
	x0, y0, x1, y1 = locations
	D = ii(gray_img,x1,y1)
	C = ii(gray_img,x0,y1)
	B = ii(gray_img,x1,y0)
	A = ii(gray_img,x0,y0)

	diff = D - C - B + A
	return diff

def partition_image(gray_img):
	"""Splits image into 3x3 windows
	:param gray_img: array of pixels
	:return: list of tuples where each tuple = (top left corner of window, bottom right corner of window)
	"""
	width, height = gray_img.shape
	x = width/3
	y = height/3
	block1 = (0,0,x-1,y-1)
	block2 = (x,0,2*x-1,y-1)
	block3 = (2*x,0,3*x-1,y-1)
	block4 = (0,y,x-1,2*y-1)
	block5 = (x,y,2*x-1,2*y-1)
	block6 = (2*x,y,3*x-1,2*y-1)
	block7 = (0,2*y,x-1,3*y-1)
	block8 = (x,2*y,2*x-1,3*y-1)
	block9 = (2*x,2*y,3*x-1,3*y-1)

	return [block1, block2, block3,
			block4, block5, block6,
			block7, block8, block9]

def feat_two_rectangles(gray_img, block_num):
	"""Calculates integral images for two windows A and B then finds their difference
	:param gray_img: array of pixels
	:param block_num: tuple (block's upper left corner, block's bottom right corner)
	:return: I(B) - I(A) where I() = integral image
	"""
	half_x = (block_num[2]-block_num[0])/2 - 1
	A = (block_num[0], block_num[1], half_x, block_num[3])
	B = (half_x + 1, block_num[1], block_num[2], block_num[3])

	A_sum = integralImage(gray_img,A)
	B_sum = integralImage(gray_img,B)
	return (B_sum - A_sum)

def feat_three_rectangles_horizontal(gray_img, block_num, height1=24, height2=14):
	"""Calculates integral images for three windows A, B, and C
	:param gray_img: array of pixels
	:param block_num: tuple (block's upper left corner, block's bottom right corner)
	:return: I(B) - (I(A) + I(C)) where I() = integral image
	"""

	A = (block_num[0], block_num[1], block_num[2], block_num[13])
	B = (block_num[0], block_num[1]+height1+1, block_num[2],  block_num[1]+height1+height2)
	C = (block_num[0], block_num[1]+height1+height2+1, block_num[2], block_num[3])

	A_sum = integralImage(gray_img,A)
	B_sum = integralImage(gray_img,B)
	C_sum = integralImage(gray_img,C)
	return (B_sum - (A_sum + C_sum))

def feat_three_rectangles_vertical(gray_img, block_num, width1=52, width2=24):
	"""Calculates integral images for three windows A, B, and C
	:param gray_img: array of pixels
	:param block_num: tuple (block's upper left corner, block's bottom right corner)
	:return: I(B) - (I(A) + I(C)) where I() = integral image
	"""

	A = (block_num[0], block_num[1], block_num[0]+width1, block_num[3])
	B = (block_num[0]+width1+1, block_num[1], block_num[0]+width1+width2, block_num[3])
	C = (block_num[0]+width1+width2+1, block_num[1], block_num[2], block_num[3])

	A_sum = integralImage(gray_img,A)
	B_sum = integralImage(gray_img,B)
	C_sum = integralImage(gray_img,C)
	return (B_sum - (A_sum + C_sum))

def feat_four_rectangles(gray_img, block_num):
	"""Calculates integral images for four windows in a square [A, B][C, D]
	:param gray_img: array of pixels
	:param block_num: tuple (block's upper left corner, block's bottom right corner)
	:return: (I(A) + I(D)) - (I(B) + I(C)) where I() = integral image
	"""
	half_x = (block_num[2] - block_num[0])/2 - 1
	half_y = (block_num[3] - block_num[1])/2 - 1

	A = (block_num[0], block_num[1], half_x, half_y)
	B = (half_x + 1, block_num[1], block_num[2], half_y)
	C = (block_num[0], half_y + 1, half_x, block_num[3])
	D = (half_x + 1, half_y + 1, block_num[2], block_num[3])

	A_sum = integralImage(gray_img,A)
	B_sum = integralImage(gray_img,B)
	C_sum = integralImage(gray_img,C)
	D_sum = integralImage(gray_img,D)
	return ((A_sum + D_sum) - (B_sum + C_sum))

# TO DO:
# function for windowing for test
# function to output 4 corners of box to mark image
# report
  
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
		alpha = (.5) * np.log((1-error)/error)
	return alpha

def normalization_constant(dists):
	"""Calculates constant to normalize distribution so that the integral sums to 1
	:param dists: List of distribution values
	:return: constant value
	"""
	normalization = sum(dists)
	return normalization

def get_gray_imgs(pos_filepath, neg_filepath):
	"""Labels and converts images to gray scale
	:param pos_filepath: path of directory containing all pos images
	:param neg_filepath: path of directory containing all neg images
	:return: list of tuples(gray image, label)
	"""
	pos_images = load_data(pos_filepath, 1)
	neg_images = load_data(neg_filepath, -1)
	images = pos_images + neg_images
	gray_imgs = list()
	for image in images:
		gray_imgs.append((read_image(image[0]), image[1]))
	return gray_imgs
  
#Call all features in here
def weak_learner(gray_imgs, features, labels, distribution):
	"""Creates decision trees for each feature and each block and calculates training error. Selects the best model for a specific feature and block based on lowest training error.
	:param gray_imgs: list where each element is an array of pixels
	:param features: list of function names to caculate features
	:param labels: list of 1's or -1's
	:param distribution: list of weights for each image
	:return: tuple of (best_model, best_block, best_feature, lowest_error_rate, correctly_classified)
	"""
	integral_images_dict = dict()
	start1 = time.time()
	for gray_img in gray_imgs:
		blocks = partition_image(gray_img)
		for i, feature in enumerate(features):
			for j, block in enumerate(blocks):
				key_img = (i,j)
				diff = feature(gray_img, block)
				if key_img in integral_images_dict:
					integral_images_dict[key_img].append(diff)
				else:
					integral_images_dict[key_img] = [diff]
	end1= time.time()
	print "time of triple loop is:", ((end1 - start1)/60), "min"
	best_feature = []
	correctly_classified = []
	lowest_error_rate = 1.0
	best_model = []
	best_block = []

	start2 = time.time()
	for k, v in integral_images_dict.items():
		X = v
		X_list = [[item] for item in X]
		clf1 = tree.DecisionTreeClassifier(max_depth = 1)
		clf = clf1.fit(X_list, labels)
		predictions = clf.predict(X_list)
		#print predictions.tolist()
		#print predictions.tolist()[0]
		#print type(predictions), type(predictions.tolist()), type(predictions.tolist()[0])
		incorrectly_classified = [x[0] != x[1] for x in zip(predictions.tolist(), labels)]
		error_rate = sum([x[0]*x[1] for x in zip(distribution, incorrectly_classified)])
		#print v
		print "error rate of key", k, "is", error_rate

		if error_rate < lowest_error_rate:
			best_feature = features[k[0]]
			best_block = k[1]
			lowest_error_rate = error_rate
			best_model = clf
			correctly_classified = [x[0] == x[1] for x in zip(predictions, labels)]
	end2 = time.time()
	print "time of classification loop is:", ((end2 - start2)/60), "min"
	return (best_model, best_block, best_feature, lowest_error_rate, correctly_classified)
  
def adaboost_train(pos_filepath, neg_filepath):
	"""Performs adaboost on training set
	:param pos_filepath: directory of positive files
	:param neg_filepath: directory of negative files
	:return: tuple of (alphas, best_models, best_blocks, best_features) where alphas are the weights of the models
	"""
	images = get_gray_imgs(pos_filepath, neg_filepath)

	gray_imgs = [x[0] for x in images]
	labels = [x[1] for x in images]

	n = len(gray_imgs)
	error = 0

	dists = [1.0/n for i in range(n)]

	alphas = list()
	correctly_classified = list()
	error_counter = 0

	T = 3

	features = [feat_two_rectangles, feat_three_rectangles, feat_four_rectangles]
	best_models = list()
	best_features = list()
	best_blocks = list()

	for t in range(T):
		best_model, best_block, best_feature, lowest_error_rate, correctly_classified = weak_learner(gray_imgs, features, labels, dists)
		print best_model, best_block, best_feature, lowest_error_rate, correctly_classified
		print lowest_error_rate
		best_models.append(best_model)
		best_features.append(best_feature)
		best_blocks.append(best_block)
		alpha = calculate_alpha(lowest_error_rate)
		alphas.append(alpha)
		for i in range(n):
			if correctly_classified[i] == True:
				dists[i] = dists[i]*np.exp(-alpha)  # update distributions for each image based on if they are correctly classified
			else:
				dists[i] = dists[i]*np.exp(alpha)
		normalization = normalization_constant(dists)
		dists = [x/normalization for x in dists]
		print "distributution", dists
	return alphas, best_models, best_blocks, best_features

def calculate_final_hypothesis(gray_img, alphas, best_models, best_blocks, best_features):
	"""Calculates the classification for a test image
	:param gray_img: array of pixel values
	:param alphas: list where the elements are weights wegihts of the models
	:param best_models: list of models selected by adaboost
	:param best_blocks: list of blocks selected by adaboost
	:param best_features: list of features selected by adaboost
	"""
	T = range(len(alphas))
	blocks = partition_image(gray_img)

	haar = [best_features[i](gray_img, blocks[best_blocks[i]]) for i in range(len(best_blocks))]
	haar = [[x] for x in haar]
	predictions = [best_models[i].predict(haar[i]) for i in range(len(best_models))]

	products = [x[0]*x[1] for x in zip(predictions, alphas)]

	classification = np.sign(sum(products))
	return classification

# def adaboost_tests(alphas, pos_test_directory, neg_test_directory):
#     error = 0
#     gray_imgs = get_gray_imgs(pos_test_directory, neg_test_directory)
#     for gray_img in gray_imgs:
#         prediction = adaboost_test(alphas, gray_img)
#         error += calculate_error(prediction, gray_img[1])
#     return error

def test_training_images(pos_testpath, neg_testpath, trained_model):
	""" test files in test directory
	:param pos_testpath:
	:param neg_testpath:
	:param trained_model:
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

		prediction = calculate_final_hypothesis(gray_img, trained_model[0],trained_model[1],trained_model[2],trained_model[3])

		if prediction == label and label == 1:
			pos_acc += 1.0
		if prediction == label and label == -1:
			neg_acc += 1.0

	print "positive accuracy", pos_acc/positive_counter
	print "negative accuracy", neg_acc/negative_counter
	print "overall accuracy", (pos_acc + neg_acc)/(positive_counter + negative_counter)
	return

def save_params(alphas, best_models, best_blocks, best_features):
	# saving parameters to files
	pickle.dump(alphas, open("save_alphas.p", "wb"))
	pickle.dump(best_blocks, open("save_blocks.p", "wb"))
	pickle.dump(best_models, open("save_models.p", "wb"))
	pickle.dump(best_features, open("save_features.p", "wb"))

	return "Parameters saved!"

def load_params(alpha_file='save_alphas.p', model_file='save_models.p', block_file='save_blocks.p', feature_file='save_features.p'):
	test_alphas = pickle.load(open(alpha_file, "rb"))
	test_blocks = pickle.load(open(block_file, "rb"))
	test_models = pickle.load(open(model_file, "rb"))
	test_features = pickle.load(open(feature_file, "rb"))

	return test_alphas, test_models, test_blocks, test_features

def main():
	red = adaboost_train(pos_filepath, neg_filepath)
	gray_img = read_image(imagefile)
	calculate_final_hypothesis(gray_img, red[0],red[1],red[2],red[3])

	test_training_images(pos_testpath, neg_testpath, red)

	gray_img = read_image(imagefile2)
	calculate_final_hypothesis(gray_img, red[0],red[1],red[2],red[3])

	gray_img = read_image(imagefile3)
	calculate_final_hypothesis(gray_img, red[0],red[1],red[2],red[3])

	t = time.time()
	gray_img = read_image(imagefile4)
	calculate_final_hypothesis(gray_img, red[0],red[1],red[2],red[3])
	print time.time() - t
	return

if __name__ == '__main__':
	main()

