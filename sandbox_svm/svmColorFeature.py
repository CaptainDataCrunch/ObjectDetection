import multiprocessing
import numpy as np
import skimage
from skimage import io



def hsv_to_feature(hsv,N,C_h,C_s,C_v):
  """ Takes an hsv picture and returns a feature vector for it.
  The vector is built as described in the paper 'Machine Learning Attacks Against the Asirra CAPTCHA' """
  res = np.zeros((N,N,C_h,C_s,C_v))
  cell_size= 250/N
  h_range = np.arange(0.0,1.0,1.0/C_h)
  h_range = np.append(h_range,1.0)
  s_range = np.arange(0.0,1.0,1.0/C_s)
  s_range = np.append(s_range,1.0)
  v_range = np.arange(0.0,1.0,1.0/C_v)
  v_range = np.append(v_range,1.0)
  for i in range(N):
    for j in range(N):
      cell= hsv[i*cell_size:i*cell_size+cell_size,j*cell_size:j*cell_size+cell_size,:]
      # check for h
      for h in range(C_h):
        h_cell = np.logical_and(cell[:,:,0]>=h_range[h],cell[:,:,0]<h_range[h+1])
        for s in range(C_s):
          s_cell = np.logical_and(cell[:,:,1]>=s_range[s],cell[:,:,1]<s_range[s+1])
          for v in range(C_v):
            v_cell = np.logical_and(cell[:,:,2]>=v_range[v],cell[:,:,2]<v_range[v+1])
            gesamt = np.logical_and(np.logical_and(h_cell,s_cell),v_cell)
            res[i,j,h,s,v] = gesamt.any()
  return np.asarray(res).reshape(-1)

def build_color_featurevector(pars):
  """ Takes an image file and the parameters of the feature vector and builds such a vector"""
  pixels,N,C_h,C_s,C_v =pars
  #rgb_bild = file_to_rgb(filename)
  #assert (rgb_bild.shape[2]==3)
  return hsv_to_feature(skimage.color.rgb2hsv(pixels),N,C_h,C_s,C_v)

def build_color_featurematrix(file_list,N,C_h,C_s,C_v):
    """ Builds the feature matrix of the jpegs in file list
    return featurematrix where the i-th row corresponds to the feature in the i-th image of the file list"
    """
    pool = multiprocessing.Pool()
    x = [(f,N,C_h,C_s,C_v) for f in file_list]
    res = pool.map(build_color_featurevector,x)
    return np.array(res)