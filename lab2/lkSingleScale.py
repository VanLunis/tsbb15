import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from calcLp2d import lp2Dfilter
from gradCalc import gradCalc
from toGray import rgb2gray
from tEstimate import tEstimateAll
from eEstimate import eEstimateAll
from interpolate import interpolImage
from interpolateAll import fullInterpol

def lkEquation(I, J, sigmaGrad, sigmaT, ksizeGrad, ksizeT):
    (dfGrad, lp2dGrad)      = lp2Dfilter(ksizeGrad,sigmaGrad)
    (dfT, lp2dT)            = lp2Dfilter(ksizeT, sigmaT)
    (Ig, Jg, Jgdx, Jgdy)    = gradCalc(I, J, dfGrad, lp2dGrad)
    (T11, T12, T22)         = tEstimateAll(Jgdx, Jgdy, lp2dT)
    (ex, ey)                = eEstimateAll(Ig, Jg, Jgdx, Jgdy, lp2dT)
# Tensor and error for a region around all points. Needs gaussfilt?

    matrisX = np.multiply(T22,ex) - np.multiply(T12,ey)
    matrisY = np.multiply(T11,ey) - np.multiply(T12,ex)

    detT = T11*T22 - np.power(T12,2)
    #detT[detT == 0] = 0.1 #To avoid div/0
    detTinv = np.divide(np.ones(detT.shape),detT)

    deltaX = np.multiply(detTinv,matrisX)
    deltaY = np.multiply(detTinv,matrisY)

    V = np.empty([I.shape[0], I.shape[1], 2])
    V[:,:,0] = deltaX
    V[:,:,1] = deltaY


    #interpolation test
    JNew = fullInterpol(J, deltaX, deltaY)

    jError = np.sum(np.absolute(I-J))
    jNewError = np.sum(np.absolute(I-JNew))

    return V, JNew, jError, jNewError
