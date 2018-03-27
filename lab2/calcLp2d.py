import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def lp2Dfilter(ksize,sigma):
    if sigma <= 0:
        sigma = 0.1

    lp = np.atleast_2d(np.exp(-0.5 * np.square(np.arange(-ksize,ksize+1,1)/sigma)))

    lp2D = conv2(lp, np.transpose(lp))
    lp2D = lp2D/np.sum(lp2D)
    df = np.atleast_2d(-1.0/np.square(sigma) * np.arange(-ksize,ksize+1,1) * lp2D)

    return (df, lp2D)
