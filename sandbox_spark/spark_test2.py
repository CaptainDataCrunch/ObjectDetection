

import os
import sys
import time

sys.path.append("/Users/vincentpham/CaptainDataCrunch-/main/")

import glob
import cv2
from adaboostFeatures2 import *
from adaboostImages import *
from adaboostData import *


# Path for spark source folder
os.environ['SPARK_HOME']="/Users/vincentpham/Downloads/spark-1.5.2/"

# Append pyspark  to Python Path
sys.path.append("/Users/vincentpham/Downloads/spark-1.5.2/python/")
try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

# Initialize SparkContext
sc = SparkContext('local')
# 10 points

#pos_filepath = '/Users/vincentpham/Desktop/tester/positive10_resize/'
#data = sc.wholeTextFiles(pos_filepath)



neg_filepath = '/Users/vincentpham/Desktop/tester/negative10_resized/'
pos_filepath = '/Users/vincentpham/Desktop/tester/positive10_resized/'

grey_imgs = get_gray_imgs(pos_filepath,neg_filepath)
start = time.time()
img_par = sc.parallelize(grey_imgs)
#img_par.count()

img_par = img_par.map(lambda x: (x[0])).zipWithIndex()

#blue = red.map(lambda x: (x[1],x[0])).flatMapValues(lambda x: x).zipWithIndex()
#range = blue.map(lambda x: x[0][0], x[0][1])

#feat_width = 24
#feat_height = 28
#x0 = 0
#y0 = 0
#x1 = feat_width
#y1 = feat_height
#coords = (x0, y0, x1, y1)


#blue1 = sc.parallelize([1,2,3,4])
#coords = sc.parallelize([(0,0,24, 28), (30, 30, 54, 58)])
#feature = sc.parallelize([feat_two_rectangles])

def get_all_coords(feat_height=28, feat_width=24, width=150, height=120, increment=5):
	x0 = 0
	y0 = 0
	x1 = feat_width
	y1 = feat_height
	all_coords = list()
	while x1 < width:
		while y1 < height:
			coords = (x0, y0, x1, y1)
			all_coords.append(coords)
			y0 += increment
			y1 += increment
		x0 += increment
		x1 += increment
		y0 = 0
		y1 = feat_height
	return all_coords

all_coords = get_all_coords()
coords = sc.parallelize(all_coords)

img_par2 = img_par.cartesian(coords)

red = img_par2.map(lambda (x,y): (y, [feat_two_rectangles(x[0], y)]))
red2 = red.reduceByKey(lambda x,y : x + y).collect()

feat_dict = [{(feat_two_rectangles, x[0]):x[1] for x in red2}][0]
end = time.time()

print "parallel finished in", ((end - start) / 60), "min"


# [(0, (0, 0, 24, 28), -6348),
#  (0, (30, 30, 54, 58), -585),
#  (1, (0, 0, 24, 28), 8777),
#  (1, (30, 30, 54, 58), 4347),
#  (2, (0, 0, 24, 28), -5894),
#  (2, (30, 30, 54, 58), -4614)]


#parallel finished in 14.0016122023 min
