CaptainDataCrunch
===============
Our mission is to detect and classify cars in a video feed in real time.
Because real time classification requires very fast computations, we make use of 
adaptive boosting that was originally used for face detection.

### Data Description

The [data](http://www.robots.ox.ac.uk/~vgg/data3.html) used in this project are two sets of images from the University of Oxford. 
One set of images contains cropped images of backs of cars with 1,155 images. 
The other set contains images with no cars with 585 images.

### Background and Motivation

A lot of classification and object detection algorithms can achieve high accuracy in many contexts.
Usually, these algorithms are not applied in real-time. 
Being able to achieve real-time classification has many applications including face, car, pedestrian, and other object detection.

One such method is a boosting method called AdaBoost (Adaptive Boosting). The general idea is that a set of *''weak learners''* 
(usually classifiers that produce results slightly better than chance) 
put together in series can produce a very strong classifier with high accuracy. A ''weak learner'' 
is defined as any classification algorithm that has an accuracy slightly better than random guessing. 
One of the main ideas of AdaBoost is to focus on the misclassifications and to focus more attention to 
them rather than the correctly classified items in a training set.

In the end, the final classification is the *sign* of a summation of weights corresponding to each ''weak learner''.

One useful tutorial for learning about AdaBoost can be found [here](http://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf). 
Viola and Jones also introduces a nice framework for working with cascades in Adaboost found [here](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf).

### Requirements

#####Python:

We are only working with Python for coding as for now. 

#####OpenCV : 

Because our data are images, we have utilized [OpenCV](http://opencv.org/downloads.html), a package in Python geared towards 
computer vision and image processing. OpenCV was used to load and transform the images to grayscale. We had difficulties getting 
OpenCV to install. 

If you are using Anaconda, this worked for us:

    conda install -c https://conda.binstar.org/menpo opencv


#####SKlearn:

We also used the [Sklearn](http://scikit-learn.org/stable/install.html), a popular package for machine learning, to create the decision tree classifiers. 
Installation instruction can be found in the link above if not already installed.
