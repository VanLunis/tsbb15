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

# Load lab data
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





#Get startingpoints to trackLK : x = col, y = row
maxes, bestMaxes, row, col = harrisMax(chess1, 6,3,[10,10], 0.05, 20000000, 5)
plt.figure()
plt.imshow(chess1, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')
# 1-2
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess1, chess2, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess2, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')


# 2-3
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess2, chess3, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess3, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')


# 3-4
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess3, chess4, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess4, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')


# 4-5
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess4, chess5, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess5, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')



# 5-6
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess5, chess6, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess6, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')



# 6-7
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess6, chess7, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess7, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')


# 7-8
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess7, chess8, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess8, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')


# 8-9
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess8, chess9, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess9, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')


# 9-10
for pointIndex in range(0,len(row)):
    dEst = trackLK(chess9, chess10, col[pointIndex], row[pointIndex], 20, 20, 11, 0)
    row[pointIndex] = row[pointIndex] + dEst[1]
    col[pointIndex] = col[pointIndex] + dEst[0]

plt.figure()
plt.imshow(chess10, cmap='gray')
plt.autoscale(False)
plt.plot(col,row, 'ro')


plt.show()
