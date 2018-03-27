import numpy as np

def rgb2gray(rgbIm):
    return np.dot(rgbIm[...,:3], [0.299, 0.587, 0.114])
