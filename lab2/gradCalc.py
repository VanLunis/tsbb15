import numpy as np
import scipy
from scipy.signal import fftconvolve as conv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def gradCalc(I, J, df, lp2D):
    Ig = conv2(I, lp2D, mode='same')
    Jg = conv2(J, lp2D, mode='same')
    Jgdx = conv2(J, df, mode='same')
    Jgdy = conv2(J, np.transpose(df), mode='same')

    return (Ig, Jg, Jgdx, Jgdy)
