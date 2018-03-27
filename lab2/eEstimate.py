import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import fftconvolve as conv2

def eEstimateAll(Ig, Jg, Jgdx, Jgdy, lp2D):
    diff = Ig-Jg

    ex = np.multiply(diff,Jgdx)
    ey = np.multiply(diff,Jgdy)

    ex = conv2(ex, lp2D, mode='same')
    ey = conv2(ey, lp2D, mode='same')

    return (ex, ey)
