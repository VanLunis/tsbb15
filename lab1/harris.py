import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from orientTensor import calcOrientTensor

def calcHarris(Im, gradKsize, gradSigma, window_size, kappa):
    T11, T12, T22 = calcOrientTensor(Im, gradKsize, gradSigma, window_size);
    Ch = np.multiply(T11,T22) - np.multiply(T12,T12) -kappa*(T11+T22)
    return Ch

def cornerThresh(Im, gradKsize, gradSigma, window_size, kappa, thresh):
    Ch = calcHarris(Im, gradKsize, gradSigma, window_size, kappa)
    threshIm = np.multiply((Ch > thresh),Ch)
    return threshIm


def harrisMax(Im, gradKsize, gradSigma, window_size, kappa, thresh, numToTrack):
    threshIm = cornerThresh(Im, gradKsize, gradSigma, window_size, kappa, thresh)
    areaSize = 5
    #img_max = scipy.signal.order_filter(threshIm, np.ones((3,3)), 9-1)
    #[row, col] = np.nonzero(threshIm == img_max)

    img_max = filters.maximum_filter(threshIm, areaSize)
    maxima = (threshIm == img_max)
    img_min = filters.minimum_filter(threshIm, areaSize)
    diff = ((img_max- img_min) > thresh)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    maxes = np.zeros(threshIm.shape)
    '''
    for dy, dx in slices:
        x_center = (dx.start + dx.stop -1)/2
        col.append(x_center)
        y_center = (dy.start + dy.stop -1 )/2
        row.append(y_center)
    return row[0:numToTrack], col[0:numToTrack]
    '''
    for dy, dx in slices:
        x_center = (dx.start + dx.stop -1)/2
        y_center = (dy.start + dy.stop -1 )/2
        maxes[int(round(y_center)), int(round(x_center))] = 1;
    indices = (-maxes).argpartition(numToTrack, axis=None)[:numToTrack]
    row, col = np.unravel_index(indices, maxes.shape)
    bestMaxes =  np.zeros(threshIm.shape)
    bestMaxes[row, col] = 1
    return maxes, bestMaxes, row, col
