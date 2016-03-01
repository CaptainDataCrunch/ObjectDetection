import cv2
from adaboostTest2 import *
from adaboostData import *
from adaboostImages import *
import numpy as np

import warnings
warnings.filterwarnings("ignore")

best_alphas, best_models, best_blocks, best_features = load_params()

# face_cascade = cv2.CascadeClassifier('cars.xml')
vc = cv2.VideoCapture('cardetection.mp4')
frameRate = vc.get(5) #frame rate
print frameRate
if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False
while rval:
    frameID = vc.get(1) # get current frame number
    # print frameID
    rval, frame = vc.read()

    # print np.round(frameID) % np.floor(frameRate)
    if np.round(frameID) % np.floor(frameRate) == 0:
        test_frame = frame[220:340,325:475] # test for region with car
        # test_frame = frame[0:120, 0:150] # test for region with no car
        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

        classification = calculate_final_hypothesis(test_frame, best_alphas, best_models, best_blocks, best_features)
    # car detection.
    # cars = face_cascade.detectMultiScale(frame, 1.1, 2)
    # ncars = 0
    # for (x,y,w,h) in cars:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    #     ncars = ncars + 1
    #
    # show result
        if classification == 1.0:
            cv2.rectangle(frame,(325,220),(325+150,220+150),(0,255,0),2)
    cv2.imshow("Result",frame)
    cv2.waitKey(1)
vc.release()