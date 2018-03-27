import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import RectBivariateSpline as RectIntp

def fullInterpol(Im, deltaX, deltaY):
    #newIm = intepMap(Im, [du[0], du[1]], mode='nearest')

    rows, cols = Im.shape
    intepOrder = 3 #Interpolation order
    interpolator = RectIntp(np.arange(rows), np.arange(cols), Im, kx=intepOrder, ky=intepOrder)

    xGrid, yGrid = np.meshgrid(np.arange(cols), np.arange(rows))

    gx = xGrid + deltaX
    gy = yGrid + deltaY

    newIm = interpolator(gy, gx, grid=False)
    return newIm
