import time
import numpy as np
from sklearn import cross_validation
from sklearn import svm


def classify_color_feature(F,y):
  start = time.time()
  clf = svm.SVC(kernel='rbf',gamma=0.001)
  scores = cross_validation.cross_val_score(clf, F, y, cv=5,n_jobs=-1)
  time_diff = time.time() - start
  print "Accuracy: %.1f  +- %.1f   (calculated in %.1f seconds)"   % (np.mean(scores)*100,np.std(scores)*100,time_diff)

