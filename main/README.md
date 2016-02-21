#### Files:

###### adaboostTrain.py
```python
def calculate_error(prediction, label):
'''Calculates error of classification'''
      
def calculate_alpha(error):
'''Calculates weight used to update the distribution'''
      
def normalization_constant(dists):
'''Calculates constant to normalize distribution so that the integral sums to 1'''

def get_feature_values(gray_imgs, features):
'''Obtains the integral images values for a given set of features and images'''

def weak_learner(gray_imgs, integral_images_dict, features, labels, distribution):
'''Selects the best model for a specific feature and block based on lowest training error'''

def adaboost_train(pos_filepath, neg_filepath, T=3):
'''Performs adaboost on training set'''
```

###### adaboostTest.py
```python
def calculate_final_hypothesis(gray_img, alphas, best_models, best_blocks, best_features):
'''Calculates the classification for a test image'''

def test_training_images(pos_testpath, neg_testpath, trained_model):
'''test files in test directory'''
```

###### adaboostData.py
```python
def load_data(filepath, label_value):
'''Loads in image and label'''

def save_params(alphas, best_models, best_blocks, best_features):

def load_params(alpha_file='save_alphas.p', model_file='save_models.p',
                block_file='save_blocks.p', feature_file='save_features.p'):
```

###### adaboostImages.py
```python
def read_image(imagefile):
'''Change image to grayscale'''

def get_gray_imgs(pos_filepath, neg_filepath):
'''Labels and converts images to gray scale'''

def s(gray_img,x,y):
'''Cumulative row sum to calculate integral image'''

def ii(gray_img,x,y):
'''Cumulative column sum to calcualte integral image'''

def integralImage(gray_img, locations):
'''Calculates integral image to compute rectangle features'''

def partition_image(gray_img):
'''Splits image into 3x3 windows'''
```

###### adaboostFeatures.py
```python
def feat_two_rectangles(gray_img, block_num):
'''Calculates integral images for two windows A and B then finds their difference'''

def feat_three_rectangles(gray_img, block_num):
'''Calculates integral images for three windows A, B, and C'''

def feat_four_rectangles(gray_img, block_num):
'''Calculates integral images for four windows in a square [A, B][C, D]'''
```

###### runAdaboost.py
```python
def main():
```
