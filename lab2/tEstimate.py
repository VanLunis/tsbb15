import numpy as np
import scipy
import math
from scipy.signal import fftconvolve as conv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def tEstimateAll(dx, dy, lp2D):
    # Tensor for a region around all points
    T = np.empty((3,1))

    T11 = np.multiply(dx,dx)
    T12 = np.multiply(dx,dy)
    T22 = np.multiply(dy,dy)

    T11 = conv2(T11, lp2D, mode='same')
    T12 = conv2(T12, lp2D, mode='same')
    T22 = conv2(T22, lp2D, mode='same')

    #Tmat = np.matrix([[T11, T12], [T12, T22]])
    #Tmat = np.empty((2,2))
    #Tmat[0,0] = T[0]
    #Tmat[1,0] = T[1]
    #Tmat[0,1] = T[1]
    #Tmat[1,1] = T[2]

    return(T11, T12, T22)
