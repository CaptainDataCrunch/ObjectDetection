import cv2
import numpy as np
import glob
from sklearn import tree
import time

imagefile = '/Users/vincentpham/Desktop/tester/image_0006.jpg'
imagefile2 = '/Users/vincentpham/Desktop/tester/image_0467.jpg'
imagefile3 = '/Users/vincentpham/Desktop/tester/image_0294.jpg'
imagefile4 = '/Users/vincentpham/Desktop/tester/image_0078.jpg'


neg_filepath = '/Users/vincentpham/Desktop/tester/negative/'
pos_filepath = '/Users/vincentpham/Desktop/tester/positive/'

def load_data(filepath, label_value):
    """
    loads in image and label
    Return: list of tuples(ndarray, label)
    """
    files = glob.glob(filepath + '*.jpg')
    labels = [label_value for i in range(len(files))]
    return zip(files, labels)

def readImage(imagefile):
    # read in image
    img = cv2.imread(imagefile)

    # change image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img
  
def s(gray_img,x,y):
    sums = 0
    if y == -1:
        return sums
    else:
        sums += gray_img[x][y] + s(gray_img,x,y-1)
    return sums

def ii(gray_img,x,y):
    sums = 0
    if x == -1:
        return sums
    else:
        sums += ii(gray_img,x-1,y) + s(gray_img,x,y)
    return sums
  
def integralImage(gray_img, locations):
    x0, y0, x1, y1 = locations
    D = ii(gray_img,x1,y1)
    C = ii(gray_img,x0,y1)
    B = ii(gray_img,x1,y0)
    A = ii(gray_img,x0,y0)

    diff = D - C - B + A
    return diff

def partition_image(gray_img):
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
    '''two windows A and B
    B - A
    '''
    half_x = (block_num[2]-block_num[0])/2 - 1
    A = (block_num[0], block_num[1], half_x, block_num[3]) 
    B = (half_x + 1, block_num[1], block_num[2], block_num[3])
    
    A_sum = integralImage(gray_img,A)
    B_sum = integralImage(gray_img,B)
    return (B_sum - A_sum)

def feat_three_rectangles(gray_img, block_num):
    '''three windows A, B, and C
    B - (A + C)
    '''
    third_x = (block_num[2]-block_num[0])/3 - 1
    
    A = (block_num[0], block_num[1], third_x, block_num[3])
    B = (third_x+1, block_num[1], 2*third_x,  block_num[3])
    C = (2*third_x+1, block_num[1], block_num[2], block_num[3])
    
    A_sum = integralImage(gray_img,A)
    B_sum = integralImage(gray_img,B)
    C_sum = integralImage(gray_img,C)
    return (B_sum - (A_sum + C_sum))

def feat_four_rectangles(gray_img, block_num):
    '''four windows in a square [A, B][C, D]
    (A + D) - (B + C)
    '''
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
    if prediction == label:
        return 0
    else:
        return 1

def calculate_alpha(error):
    alpha = 0
    if error != 0:
        alpha = (.5) * np.log((1-error)/error)
    return alpha
                
def normalization_constant(dists):
    normalization = sum(dists)
    return normalization

def get_gray_imgs(pos_filepath, neg_filepath):
    pos_images = load_data(pos_filepath, 1)
    neg_images = load_data(neg_filepath, -1)
    images = pos_images + neg_images
    gray_imgs = list()
    for image in images:
        gray_imgs.append((readImage(image[0]),image[1]))
    return gray_imgs
  
#Call all features in here
def weak_learner(gray_imgs, features, labels, distribution):
  #add distribution part
    
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
  
def main():
    red = adaboost_train(pos_filepath, neg_filepath)
    gray_img = readImage(imagefile)
    calculate_final_hypothesis(gray_img, red[0],red[1],red[2],red[3])

    gray_img = readImage(imagefile2)
    calculate_final_hypothesis(gray_img, red[0],red[1],red[2],red[3])

    gray_img = readImage(imagefile3)
    calculate_final_hypothesis(gray_img, red[0],red[1],red[2],red[3])

    gray_img = readImage(imagefile4)
    calculate_final_hypothesis(gray_img, red[0],red[1],red[2],red[3])
    return
                      
if __name__ == '__main__':
    main()

>>> from sklearn import tree
>>> X = [[0, 0], [1, 1],[1,0]]
>>> Y = [0, 1]
>>> clf = tree.DecisionTreeClassifier()
>>> clf = clf.fit(X, Y)
clf.predict(X).tolist()