### SVM

Ran SVM on color values of image on a small amount of training data for preliminary analsysis. Cross-validated training accuracy are

    Accuracy: 95.7  +- 3.2   (calculated in 3.4 seconds)
    Accuracy: 99.7  +- 0.4   (calculated in 9.7 seconds)
    Accuracy: 99.9  +- 0.1   (calculated in 12.2 seconds)
    Accuracy: 99.9  +- 0.1   (calculated in 10.4 seconds)
    Accuracy: 99.9  +- 0.1   (calculated in 22.7 seconds)
  
Reused most of the functions from [Martin Böschen](https://github.com/Safadurimo/cats-and-dogs/blob/master/catsdogs.ipynb) 

#### Files:

###### svmData.py
```python
def load_data(filepath, label_value):
'''Loads in image and label'''
    
def load_and_label_data(pos_filepath, neg_filepath):
'''Labels and converts images to gray scale'''
```

###### svmColorFeature.py
```python
def hsv_to_feature(hsv,N,C_h,C_s,C_v):
'''Takes an hsv picture and returns a feature vector for it.'''

def build_color_featurevector(pars):
'''Takes an image file and the parameters of the feature vector and builds such a vector'''

def build_color_featurematrix(pixels,N,C_h,C_s,C_v):
'''Builds the feature matrix of the jpegs in file list'''
```

###### svmClassify.py
```python
def classify_color_feature(F,y):
''' Get cross-validated training accuracy
```

###### runSVM.py
```python
def main():
```
