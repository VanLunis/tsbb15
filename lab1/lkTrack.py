import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from gradCalc import gradCalc
from toGray import rgb2gray
from estimateT import estimateT
from estimateE import estimateE
from interpolate import interpolImage

def trackLK(I, J, x, y, height, width, maxSteps, maxErr):
    if maxErr == 0:
        maxErr = -1
    steps = 0;
    currErr = 10000000
    dEst = 0;

    left = math.floor(x-width/2)
    top = math.floor(y-height/2)
    right = left + width
    bot = top + height

    Ig, Jg, Jgdx, Jgdy = gradCalc(I, J, 6, 3)

    #Windowed versions of all images
    JgdxWin = Jgdx[top:bot,left:right]
    JgdyWin = Jgdy[top:bot,left:right]
    IgWindowed = Ig[top:bot,left:right]
    JgWin = Jg[top:bot,left:right]

    #print('Shape of IgWin: ' + str(IgWin.shape))

    JgOld = Jg
    JgdxOld = Jgdx
    JgdyOld = Jgdy


    while ((steps < maxSteps) and (currErr > maxErr)):
        T = estimateT(Jgdx, Jgdy, x, y, [height,width])
        e = estimateE(Ig, Jg, Jgdx, Jgdy, x, y, [height,width])
        dEst = dEst + np.linalg.solve(T,e)

        #print('********Results**********')
        #print('Step: ' + str(steps))
        #print('dEst: ' + str(dEst))

        '''
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        for im, name, ax in zip((IgWindowed, Jg[top:bot,left:right]), ('I','J itr: ' + str(steps) ), axes.flat):
            ax.imshow(im, cmap='gray')
            ax.set_title(name)
        '''

        Jg, Jgdx, Jgdy = interpolImage(JgOld, JgdxOld, JgdyOld, [-dEst[0], -dEst[1]], 0, JgOld.shape[1], 0, JgOld.shape[0])

        error = np.linalg.norm(dEst)
        steps = steps + 1


        #print('********End of step***********')


    #plt.show()
    return dEst
