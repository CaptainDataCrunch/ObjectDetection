import cv2
from adaboostTest import *
from adaboostData import *
from adaboostImages import *
import numpy as np

import warnings
warnings.filterwarnings("ignore")

best_alphas, best_models, best_blocks, best_features = load_params()
SCALE = 0.666
print best_blocks
best_blocks_scaled = [tuple([int(SCALE*x) for x in item]) for item in best_blocks]

SCALE_2 = 0.333
best_blocks_scaled_2 = [tuple([int(SCALE_2*x) for x in item]) for item in best_blocks]
print best_blocks_scaled
print best_blocks_scaled_2

# face_cascade = cv2.CascadeClassifier('cars.xml')
vc = cv2.VideoCapture('cardetection2.mp4')
video = cv2.VideoWriter()
video.open('videofinal2.avi',cv2.cv.CV_FOURCC('m','p','4','v'), 20, (854,480), True)
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
    # if np.round(frameID) % np.floor(frameRate) == 0:
    if True:
        test_frame = frame[200:320,352:502] # test for region with car
        test_frame_middle_scaled = frame[200+20:320-20, 352+25:502-25]
        test_frame_middle_scaled_2 = frame[200+20+20:320-20-20, 352+25+25:502-25-25]

        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        test_frame_middle_scaled = cv2.cvtColor(test_frame_middle_scaled, cv2.COLOR_BGR2GRAY)
        test_frame_middle_scaled_2 = cv2.cvtColor(test_frame_middle_scaled_2, cv2.COLOR_BGR2GRAY)

        classification_middle = calculate_final_hypothesis(test_frame,
                                                           best_alphas,
                                                           best_models,
                                                           best_blocks,
                                                           best_features)
        classification_middle_scaled = calculate_final_hypothesis(test_frame_middle_scaled,
                                                                  best_alphas,
                                                                  best_models,
                                                                  best_blocks_scaled,
                                                                  best_features,
                                                                  scale = SCALE)
        classification_middle_scaled_2 = calculate_final_hypothesis(test_frame_middle_scaled_2,
                                                                    best_alphas,
                                                                    best_models,
                                                                    best_blocks_scaled_2,
                                                                    best_features,
                                                                    scale = SCALE_2)


    # car detection.
    # cars = face_cascade.detectMultiScale(frame, 1.1, 2)
    # ncars = 0
    # for (x,y,w,h) in cars:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    #     ncars = ncars + 1
    #
    # show result
    #     if classification == 1.0:
    #         cv2.rectangle(frame,(352,200),(325+150,220+150),(0,255,0),2)
        if classification_middle_scaled_2 == 1.0:
            if classification_middle_scaled == 1.0:
                if classification_middle == 1.0:
                    cv2.rectangle(frame,(352,200),(352+150,200+150),(0,255,0),2)
                else:
                    cv2.rectangle(frame,(352+25,200+20),(352+25+80,200+20+80),(0,255,0),2)
            else:
                cv2.rectangle(frame,(352+25+25,200+20+20),(352+25+25+50,200+20+20+40),(0,255,0),2)
    video.write(frame)
    cv2.imshow("Result",frame)
    cv2.waitKey(1)
video.release()
vc.release()