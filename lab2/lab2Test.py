# Lab2 test
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cvl_labs.lab1 import load_lab_image, get_cameraman
import cvl_labs.lab2
from scipy.ndimage.interpolation import shift as intepShift

from calcLp2d import lp2Dfilter
from gradCalc import gradCalc
from toGray import rgb2gray
from tEstimate import tEstimateAll
from eEstimate import eEstimateAll
from lkSingleScale import lkEquation
<<<<<<< HEAD
from errorDiff import imDiff, imShift
=======
from interpolate import interpolImage
from lkMultiscale import multiLKeq
>>>>>>> 61ea2e89492b662f9d0733686b1425e8f8a8ed17

I = mpimg.imread('forwardL/forwardL0.png')
J = mpimg.imread('forwardL/forwardL1.png')

sigmaGrad = 1
sigmaT = 10
ksizeGrad = 3
ksizeT = 15
numScales = 4

V, JNew, jError, jNewError =multiLKeq(I,J,numScales,sigmaGrad, sigmaT, ksizeGrad, ksizeT)

cvl_labs.lab2.gopimage(V, scale=3)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
for im, name, ax in zip((I, JNew), ('I', 'JNew'), axes.flat):
    ax.imshow(im, cmap='gray')
    ax.set_title(name)

print('Jerror: ' + str(jError))
print('jNewError: ' + str(jNewError))

#I = load_lab_image('chessboard_5.png')
#J = load_lab_image('chessboard_6.png')

<<<<<<< HEAD
cornerIm = load_lab_image('cornertest.png')
I = cornerIm
=======
#I, J, dTrue = get_cameraman()

#cornerIm = load_lab_image('cornertest.png')
#I = cornerIm

# testing with translated image
#I, J, dTrue = get_cameraman()
>>>>>>> 61ea2e89492b662f9d0733686b1425e8f8a8ed17
'''
rows = I.shape[0]
cols = I.shape[1]
J = np.empty([rows,cols])

for y in range(1,rows):
    for x in range(2,cols-1):
        J[y,x] = I[y-1,x-2]
        # d = (2,1)
'''

<<<<<<< HEAD
J = imShift(I)
=======
'''
# Show the two images I, J to compare shift
>>>>>>> 61ea2e89492b662f9d0733686b1425e8f8a8ed17
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
for im, name, ax in zip((I, J), ('I', 'J'), axes.flat):
    ax.imshow(im, cmap='gray')
    ax.set_title(name)

sigmaGrad = 3
sigmaT = 5
ksizeGrad = 8
ksizeT = 16
'''

'''

#help(cvl_labs.lab2)
V, JNew, jError, jNewError = lkEquation(I, J, sigmaGrad, sigmaT, ksizeGrad, ksizeT)
print('Jnew shape: ' + str(JNew.shape))
cvl_labs.lab2.gopimage(V, scale=5)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
for im, name, ax in zip((I, JNew), ('I', 'JNew'), axes.flat):
    ax.imshow(im, cmap='gray')
    ax.set_title(name)


print('Jerror: ' + str(jError))
print('jNewError: ' + str(jNewError))
'''

# test lk equation
'''
(dfGrad, lp2dGrad)      = lp2Dfilter(ksizeGrad,sigmaGrad)
(dfT, lp2dT)            = lp2Dfilter(ksizeT, sigmaT)
(Ig, Jg, Jgdx, Jgdy)    = gradCalc(I, J, dfGrad, lp2dGrad)
(T11, T12, T22)         = tEstimateAll(Jgdx, Jgdy, lp2dT)
(ex, ey)                = eEstimateAll(Ig, Jg, Jgdx, Jgdy, lp2dT)
# Tensor and error for a region around all points. Needs gaussfilt?

matrisX = np.multiply(T22,ex) - np.multiply(T12,ey)
matrisY = np.multiply(T12,ey) - np.multiply(T12,ex)

detT = T11*T22 - np.power(T12,2)
detT[detT == 0] = 0.00001 #To avoid div/0
detTinv = np.divide(np.ones(detT.shape),detT)

deltaX = np.multiply(detTinv,matrisX)
deltaX[deltaX > 0.0001] = 0.0001
deltaY = np.multiply(detTinv,matrisY)
deltaY[deltaY > 0.0001] = 0.0001

V = np.empty([I.shape[0], I.shape[1], 2])
V[:,:,0] = deltaX
V[:,:,1] = deltaY
'''

# Test show lp-filter and lpgrad-filter
'''
plt.figure()
plt.title('Ig')
plt.imshow(Ig, cmap='gray')
plt.figure()
plt.title('Jg')
plt.imshow(Jg, cmap='gray')
plt.figure()
plt.title('Jgdx')
plt.imshow(Jgdx, cmap='gray')
plt.figure()
plt.title('Jgdy')
plt.imshow(Jgdy, cmap='gray')
'''

#Test show deltaX, deltaY
'''
plt.figure()
plt.title('dx')
plt.imshow(dx, cmap='gray')
plt.figure()
plt.title('dy')
plt.imshow(dy, cmap='gray')
'''







plt.show()
