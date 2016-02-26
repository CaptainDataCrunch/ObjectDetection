__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

import time
from adaboostTrain import *
from adaboostTest import *

neg_filepath = '/Users/vincentpham/Desktop/tester/negative10_resized/'
pos_filepath = '/Users/vincentpham/Desktop/tester/positive10_resized/'

neg_testpath = '/Users/vincentpham/Desktop/tester/negative_test/'
pos_testpath = '/Users/vincentpham/Desktop/tester/positive_test/'

def main():
	start = time.time()
	trained_model = adaboost_train(pos_filepath, neg_filepath, T = 2)
	cp1 = time.time()
	print "alphas:", trained_model[0]
	print "training finished in:", ((cp1 - start)/60), "min"

	test_training_images(pos_testpath, neg_testpath, trained_model)
	cp2 = time.time()

	print "testing finished in:", ((cp2 - cp1)/60), "min"
	return

if __name__ == '__main__':
	main()
