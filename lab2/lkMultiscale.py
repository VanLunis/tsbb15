import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cvl_labs.lab2

from calcLp2d import lp2Dfilter
from gradCalc import gradCalc
from toGray import rgb2gray
from tEstimate import tEstimateAll
from eEstimate import eEstimateAll
from interpolate import interpolImage
from interpolateAll import fullInterpol
from lkSingleScale import lkEquation

def multiLKeq(I,J,numScales, sigmaGrad, sigmaT, kSizeGrad, kSizeT):
    Vtot = 0
    Jn = J
    jError = 0
    jNewError = 0
    for n in range(numScales, 0, -1):
        print('n= ', n)
        sc = 2 ** (n-1)
        Vn, Jn, jError, jNewError  = lkEquation(I, Jn, sc * sigmaGrad, sc * sigmaT, sc * kSizeGrad, sc * kSizeT)
        Vtot = Vtot + Vn
        #Jn = fullInterpol(J,Vn[:,:,0], Vn[:,:,1])
        cvl_labs.lab2.gopimage(Vtot, scale=3)
        plt.title(f'Vn for {n}')
    return Vtot, Jn, jError, jNewError
