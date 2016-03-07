from adaboostTest2 import *
from adaboostData import *
from adaboostImages import *
import cv2
import multiprocessing
import warnings
import numpy as np
warnings.filterwarnings("ignore")

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default

best_alphas, best_models, best_blocks, best_features = load_params()
# best_alphas = best_alphas[0:3]
# best_models = best_models[0:3]
# best_blocks = best_blocks[0:3]
# best_features = best_features[0:3]

def work(frame):
    test_frame = frame[220:340,325:475] # test for region with car
    # test_frame = frame[0:120, 0:150] # test for region with no car
    test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

    classification = calculate_final_hypothesis(test_frame, best_alphas, best_models, best_blocks, best_features)
    if classification == 1.0:
        cv2.rectangle(frame,(325,220),(325+150,220+150),(0,255,0),2)
        video.write(frame)


# face_cascade = cv2.CascadeClassifier('cars.xml')
vc = cv2.VideoCapture('cardetection.mp4')
video = cv2.VideoWriter()
video.open('videonew2.avi',cv2.cv.CV_FOURCC('m','p','4','v'), 20, (854,480), True)
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
    video.write(frame)
    # cv2.imshow("Result", frame)

    # test only certain frames
    # if np.round(frameID) % np.floor(frameRate) == 0:
    if np.round(frameID) % np.floor(30) == 0:
        test_frame = frame[220:340,325:475] # test for region with car
        # test_frame = frame[0:120, 0:150] # test for region with no car
        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

        classification = calculate_final_hypothesis(test_frame, best_alphas, best_models, best_blocks, best_features)
        if classification == 1.0:
            cv2.rectangle(frame,(325,220),(325+150,220+150),(0,255,0),2)
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
video.release()
vc.release()
cv2.destroyAllWindows()