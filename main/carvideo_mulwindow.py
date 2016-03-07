from adaboostTest2 import *
from adaboostData import *
from adaboostImages import *
import cv2
import multiprocessing
import warnings
import numpy as np
warnings.filterwarnings("ignore")


best_alphas, best_models, best_blocks, best_features = load_params()
# best_alphas = best_alphas[0:3]
# best_models = best_models[0:3]
# best_blocks = best_blocks[0:3]
# best_features = best_features[0:3]
SCALE = 0.666
print best_blocks
best_blocks_scaled = [tuple([int(SCALE*x) for x in item]) for item in best_blocks]
print best_blocks_scaled

def work(frame):
    test_frame = frame[220:340,325:475] # test for region with car
    # test_frame = frame[0:120, 0:150] # test for region with no car
    test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

    classification = calculate_final_hypothesis(test_frame, best_alphas, best_models, best_blocks, best_features)
    if classification == 1.0:
        cv2.rectangle(frame,(325,220),(325+150,220+150),(0,255,0),2)
        # video.write(frame)


# face_cascade = cv2.CascadeClassifier('cars.xml')
vc = cv2.VideoCapture('cardetection2.mp4')
video = cv2.VideoWriter()
video.open('videofinal.avi',cv2.cv.CV_FOURCC('m','p','4','v'), 20, (854,480), True)
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
    if not rval:
        break
    ### saving frames for video
    # video.write(frame)
    # cv2.imshow("Result", frame)

    # test only certain frames
    # if np.round(frameID) % np.floor(frameRate) == 0:
    if np.round(frameID) % np.floor(30) == 0:
        test_frame_middle = frame[200:320,352:502] # midpoint = 260, 427 # test for region with car
        test_frame_midleft = frame[200:320,181:331]
        test_frame_left = frame[200:320, 10:160]
        test_frame_midright = frame[200:320, 523:673]
        test_frame_right = frame[200:320, 694:844]
        test_frame_middle_scaled = frame[200+20:320-20, 352+25:502-25]
        test_frame_midleft_scaled = frame[200+20:320-20, 181+25:331-25]
        test_frame_left_scaled = frame[200+20:320-20, 10+25:160-25]
        test_frame_midright_scaled = frame[200+20:320-20, 523+25:673-25]
        test_frame_right_scaled = frame[200+20:320-20, 694+25:844-25]

        test_frame_middle = cv2.cvtColor(test_frame_middle, cv2.COLOR_BGR2GRAY)
        test_frame_midleft = cv2.cvtColor(test_frame_midleft, cv2.COLOR_BGR2GRAY)
        test_frame_left = cv2.cvtColor(test_frame_left, cv2.COLOR_BGR2GRAY)
        test_frame_midright = cv2.cvtColor(test_frame_midright, cv2.COLOR_BGR2GRAY)
        test_frame_right = cv2.cvtColor(test_frame_right, cv2.COLOR_BGR2GRAY)
        test_frame_middle_scaled = cv2.cvtColor(test_frame_middle_scaled, cv2.COLOR_BGR2GRAY)
        test_frame_midleft_scaled = cv2.cvtColor(test_frame_midleft_scaled, cv2.COLOR_BGR2GRAY)
        test_frame_left_scaled = cv2.cvtColor(test_frame_left_scaled, cv2.COLOR_BGR2GRAY)
        test_frame_midright_scaled = cv2.cvtColor(test_frame_midright_scaled, cv2.COLOR_BGR2GRAY)
        test_frame_right_scaled = cv2.cvtColor(test_frame_right_scaled, cv2.COLOR_BGR2GRAY)

        classification_middle = calculate_final_hypothesis(test_frame_middle, best_alphas, best_models, best_blocks, best_features)
        classification_midleft = calculate_final_hypothesis(test_frame_midleft, best_alphas, best_models, best_blocks, best_features)
        classification_left = calculate_final_hypothesis(test_frame_left, best_alphas, best_models, best_blocks, best_features)
        classification_midright = calculate_final_hypothesis(test_frame_midright, best_alphas, best_models, best_blocks, best_features)
        classification_right = calculate_final_hypothesis(test_frame_right, best_alphas, best_models, best_blocks, best_features)
        classification_middle_scaled = calculate_final_hypothesis(test_frame_middle_scaled, best_alphas, best_models, best_blocks_scaled, best_features)
        classification_midleft_scaled = calculate_final_hypothesis(test_frame_midleft_scaled, best_alphas, best_models, best_blocks_scaled, best_features)
        classification_left_scaled = calculate_final_hypothesis(test_frame_left_scaled, best_alphas, best_models, best_blocks_scaled, best_features)
        classification_midright_scaled = calculate_final_hypothesis(test_frame_midright_scaled, best_alphas, best_models, best_blocks_scaled, best_features)
        classification_right_scaled = calculate_final_hypothesis(test_frame_right_scaled, best_alphas, best_models, best_blocks_scaled, best_features)

        # if classification_middle == 1.0:
        #     cv2.rectangle(frame,(352,200),(352+150,200+150),(0,255,0),2)
        # if classification_midleft == 1.0:
        #     cv2.rectangle(frame,(181,200),(181+150,200+150),(0,255,0),2)
        # if classification_left == 1.0:
        #     cv2.rectangle(frame,(10,200),(10+150,200+150),(0,255,0),2)
        # if classification_midright == 1.0:
        #     cv2.rectangle(frame,(523,200),(523+150,200+150),(0,255,0),2)
        # if classification_right == 1.0:
        #     cv2.rectangle(frame,(694,200),(694+150,200+150),(0,255,0),2)
        # if classification_middle_scaled == 1.0:
        #     cv2.rectangle(frame,(352+25,200+20),(352+25+80,200+20+80),(0,255,0),2)
        # if classification_midleft_scaled == 1.0:
        #     cv2.rectangle(frame,(181+25,200+20),(181+25+80,200+20+80),(0,255,0),2)
        # if classification_left_scaled == 1.0:
        #     cv2.rectangle(frame,(10+25,200+20),(10+25+80,200+20+80),(0,255,0),2)
        # if classification_midright_scaled == 1.0:
        #     cv2.rectangle(frame,(523+25,200+20),(523+25+80,200+20+80),(0,255,0),2)
        # if classification_right_scaled == 1.0:
        #     cv2.rectangle(frame,(694+25,200+20),(694+25+80,200+20+80),(0,255,0),2)

        if classification_middle_scaled == 1.0:
            if classification_middle == 1.0:
                cv2.rectangle(frame,(352,200),(352+150,200+150),(0,255,0),2)
            else:
                cv2.rectangle(frame,(352+25,200+20),(352+25+80,200+20+80),(0,255,0),2)
        if classification_midleft_scaled == 1.0:
            if classification_midleft == 1.0:
                cv2.rectangle(frame,(181,200),(181+150,200+150),(0,255,0),2)
            else:
                cv2.rectangle(frame,(181+25,200+20),(181+25+80,200+20+80),(0,255,0),2)
        if classification_left_scaled == 1.0:
            if classification_left == 1.0:
                cv2.rectangle(frame,(10,200),(10+150,200+150),(0,255,0),2)
            else:
                cv2.rectangle(frame,(10+25,200+20),(10+25+80,200+20+80),(0,255,0),2)
        if classification_midright_scaled == 1.0:
            if classification_midright == 1.0:
                cv2.rectangle(frame,(523,200),(523+150,200+150),(0,255,0),2)
            else:
                cv2.rectangle(frame,(523+25,200+20),(523+25+80,200+20+80),(0,255,0),2)
        if classification_right_scaled == 1.0:
            if classification_right == 1.0:
                cv2.rectangle(frame,(694,200),(694+150,200+150),(0,255,0),2)
            else:
                cv2.rectangle(frame,(694+25,200+20),(694+25+80,200+20+80),(0,255,0),2)






    video.write(frame)

    # car detection.
    # cars = face_cascade.detectMultiScale(frame, 1.1, 2)
    # ncars = 0
    # for (x,y,w,h) in cars:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    #     ncars = ncars + 1
    #
    # show result

    cv2.imshow("Result", frame)
    cv2.waitKey(1)
# video.release()
vc.release()
cv2.destroyAllWindows()