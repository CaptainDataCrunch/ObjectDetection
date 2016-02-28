import json


import os
import sys
import time


sys.path.append("/Users/vincentpham/CaptainDataCrunch-/main/")

import glob
import cv2
from adaboostFeatures2 import *
from adaboostImages import *
from adaboostData import *


#pos_filepath = '/Users/vincentpham/Desktop/tester/positive10_resize/'
#data = sc.wholeTextFiles(pos_filepath)



neg_filepath = '/Users/vincentpham/Desktop/tester/negative10_resized/'
pos_filepath = '/Users/vincentpham/Desktop/tester/positive10_resized/'

grey_imgs = get_gray_imgs(pos_filepath,neg_filepath)


import pickle
images = [x[0] for x in grey_imgs]
pkl_file=open('test.txt','wb')
pickle.dump(images,pkl_file)
pkl_file.close()
#pickle.dump(listoflist,pkl_file)

orange = pickle.load(open("test.txt","rb"))
