from svmColorFeature import *
from svmClassify import *
from svmData import *

def main():
    neg_filepath = '/Users/vincentpham/Desktop/tester/carsresized/'
    pos_filepath = '/Users/vincentpham/Desktop/tester/carsbgresized/'
    images, labels = load_and_label_data(pos_filepath, neg_filepath)
    F1 = build_color_featurematrix(images,1,10,10,10)
    F2 = build_color_featurematrix(images,3,10,8,8)
    F3 = build_color_featurematrix(images,5,10,6,6)
    union = np.hstack((F1,F2,F3))


    classify_color_feature(F1,labels)
    classify_color_feature(F2,labels)
    classify_color_feature(F3,labels)

    classify_color_feature(F3,labels)

    classify_color_feature(union,labels)
    classify_color_feature(union,labels)

    # Accuracy: 95.7  +- 3.2   (calculated in 3.4 seconds)
    # Accuracy: 99.7  +- 0.4   (calculated in 9.7 seconds)
    # Accuracy: 99.9  +- 0.1   (calculated in 12.2 seconds)
    # Accuracy: 99.9  +- 0.1   (calculated in 10.4 seconds)
    # Accuracy: 99.9  +- 0.1   (calculated in 22.7 seconds)

if __name__ == "__main__":
    main()