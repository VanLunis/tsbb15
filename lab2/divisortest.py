import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cvl_labs.lab1 import load_lab_image, get_cameraman
from scipy.ndimage.interpolation import shift as intepShift

test = np.divide([18, 16, 20], [3, 2, 5])
print(test)

testmult = np.multiply([2,2,2],[2,3,4])
print(testmult)


testpower = np.power([2,2,2],[2,3,4])
print(testpower)
