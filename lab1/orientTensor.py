import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gradCalc import gradCalc
from scipy.signal import convolve2d as conv2

def calcOrientTensor(Im, gradKsize, gradSigma, window_size):
    Ig, Jg, dx, dy = gradCalc(Im, Im, gradKsize, gradSigma)


    convWindow = np.ones((window_size[0],window_size[1]), dtype=np.int)

    T11 = conv2(dx*dx,convWindow, mode='same')
    T12 = conv2(dx*dy,convWindow, mode='same')
    T22 = conv2(dy*dy,convWindow, mode='same')
    return (T11, T12, T22)
