import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cvl_labs.lab1 import load_lab_image, get_cameraman

from gradCalc import gradCalc
from toGray import rgb2gray
from estimateT import estimateT
from estimateE import estimateE
from interpolate import interpolImage
from lkTrack import trackLK
from orientTensor import calcOrientTensor
from harris import calcHarris, cornerThresh, harrisMax
from scipy.ndimage.interpolation import shift as intepShift


I, J, dTrue = get_cameraman()

chess1 = load_lab_image('chessboard_1.png')
chess2 = load_lab_image('chessboard_2.png')
chess3 = load_lab_image('chessboard_3.png')
chess4 = load_lab_image('chessboard_4.png')
chess5 = load_lab_image('chessboard_5.png')
chess6 = load_lab_image('chessboard_6.png')
chess7 = load_lab_image('chessboard_7.png')
chess8 = load_lab_image('chessboard_8.png')
chess9 = load_lab_image('chessboard_9.png')
chess10 = load_lab_image('chessboard_10.png')


#Gradcalc test
'''
Ig, Jg, Jgdx, Jgdy = gradCalc(I, J, 6, 3)

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

plt.show()
'''

# Estimate T test
'''
print('Shape of image: ' + str(I.shape))
Ig, Jg, Jgdx, Jgdy = gradCalc(I, J, 6, 3)
T = estimateT(Jgdx, Jgdy, 118, 118, [236,236])
plt.show()
'''

# Estimate e test
'''
Ig, Jg, Jgdx, Jgdy = gradCalc(I, J, 6, 3)
e = estimateE(Ig, Jg, Jgdx, Jgdy, 120, 85, [70,40])
print('e in test: ' + str(e))
'''

# interpolation function test
'''
Ig, Jg, Jgdx, Jgdy = gradCalc(I, J, 6, 3)
JNew, dxNew, dyNew = interpolImage(J, Jgdx, Jgdy, [-dTrue[0],-dTrue[1]], 0, 236, 0, 236)


fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
for im, name, ax in zip((I, JNew, J), ('I','Jnew', 'J'), axes.flat):
    ax.imshow(im, cmap='gray')
    ax.set_title(name)
plt.show()
'''
'''
plt.figure()
plt.subplot(1, 3, 1)
plt.title('I')
plt.imshow(I, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Jnew')
plt.imshow(JNew, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('J')
plt.imshow(J, cmap='gray')
'''
'''
plt.figure()
plt.subplot(1, 2, 1)
plt.title('Dx')
plt.imshow(Jgdx, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('DxNew')
plt.imshow(dxNew, cmap='gray')

plt.figure()
plt.subplot(1, 2, 1)
plt.title('Dy')
plt.imshow(Jgdy, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('DyNew')
plt.imshow(dyNew, cmap='gray')

plt.show()
'''

'''
#Tracker test
trackLK(I, J, 120, 85, 70, 40, 4, 0)
print('dTrue: ' + str(dTrue))
'''


'''
# Orient tensor test
T11, T12, T22 = calcOrientTensor(I,6,3,[10,10])
plt.figure()
plt.subplot(1, 3, 1)
plt.title('T11')
plt.imshow(T11, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('T12')
plt.imshow(T12, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('T22')
plt.imshow(T22, cmap='gray')
plt.show()
'''


#Harris response test
cornerIm = load_lab_image('chessboard_1.png')
Ch = calcHarris(cornerIm, 6,3,[10,10], 0.05)
plt.figure()
plt.imshow(cornerIm, cmap='gray')

plt.figure()
plt.imshow(Ch, cmap='gray')
plt.show()


'''
#Harris threshold test
cornerIm = load_lab_image('chessboard_1.png')
threshIm = cornerThresh(cornerIm, 6,3,[10,10], 0.05, 150000000)
plt.figure()
plt.imshow(cornerIm, cmap='gray')

plt.figure()
plt.imshow(threshIm, cmap='gray')
plt.show()
'''
'''
#*****************************BAD*****************************
# Harris point test OLD BasdjsabdgfdkjbnNONONO
row, col = harrisMax(cornerIm, 6,3,[10,10], 0.05, 150000000, 5)
print('Row: ' +str(row))
print('Col: ' +str(col))
plt.show()
#*************************************************************
'''

'''
# Harris point test NEW
maxes, bestMaxes, row, col = harrisMax(cornerIm, 6,3,[10,10], 0.05, 150000000, 5)
print('x: ' + str(col))
print('y: ' + str(row))
plt.figure()
plt.imshow(maxes, cmap='gray')
plt.figure()
plt.imshow(bestMaxes, cmap='gray')
plt.show()
'''

'''
# Harris point test NEW
maxes, bestMaxes, row, col = harrisMax(chess1, 6,3,[10,10], 0.05, 15000000, 5)
print('x: ' + str(col))
print('y: ' + str(row))
plt.figure()
plt.imshow(chess1, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')
plt.figure()
plt.imshow(maxes, cmap='gray')
plt.figure()
plt.imshow(bestMaxes, cmap='gray')
plt.show()
'''
