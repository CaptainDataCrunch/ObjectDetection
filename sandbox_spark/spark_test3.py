

import os
import sys
import time


#sys.path.append("/Users/vincentpham/CaptainDataCrunch-/main/")

import glob
from adaboostFeatures2 import *
from adaboostImages import *
from adaboostData import *
from pyspark import SparkContext
from pyspark import SparkConf

# Path for spark source folder
#os.environ['SPARK_HOME']="/Users/vincentpham/Downloads/spark-1.5.2/"

# Append pyspark  to Python Path
#sys.path.append("/Users/vincentpham/Downloads/spark-1.5.2/python/")
#try:
#    from pyspark import SparkContext
#    from pyspark import SparkConf

#    print ("Successfully imported Spark Modules")

#except ImportError as e:
#    print ("Can not import Spark Modules", e)
#    sys.exit(1)

# Initialize SparkContext
#sc = SparkContext('local')
# 10 points

#pos_filepath = '/Users/vincentpham/Desktop/tester/positive10_resize/'
#data = sc.wholeTextFiles(pos_filepath)

sc = SparkContext('local')

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

grey_imgs = pickle.load(open("test.txt","rb"))
start = time.time()
img_par = sc.parallelize(grey_imgs,4)
#img_par.count()

img_par = img_par.zipWithIndex()

s1 = time.time()
all_coords = get_all_coords()
e1 = time.time()
print "looping finished in", ((e1 - s1) / 60), "min"

s2 = time.time()

coords = sc.parallelize(all_coords)

img_par = img_par.cartesian(coords)
e2 = time.time()
print "cartesian finished in", ((e2 - s2) / 60), "min"

s3 = time.time()
img_par = img_par.map(lambda (x,y): (y, [feat_two_rectangles(x[0], y)]))
e3 = time.time()
print "function finished in", ((e3 - s3) / 60), "min"

s4= time.time()

img_par_data = img_par.reduceByKey(lambda x,y : x + y).collect()
feat_dict = [{(feat_two_rectangles, x[0]):x[1] for x in img_par_data}][0]

e4 = time.time()
print "function finished in", ((e4 - s4) / 60), "min"

end = time.time()

print "parallel finished in", ((end - start) / 60), "min"


