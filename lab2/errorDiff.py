import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cvl_labs.lab1 import load_lab_image, get_cameraman
import cvl_labs.lab2
from scipy.ndimage.interpolation import shift as intepShift

def imDiff(I,J):
    return np.sum(np.abs(I-J))

def imShift(I):
    rows = I.shape[0]
    cols = I.shape[1]
    J = np.empty([rows,cols])

    for y in range(1,rows):
        for x in range(2,cols-1):
            J[y,x] = I[y-1,x-2]
            # d = (2,1)
    return J
