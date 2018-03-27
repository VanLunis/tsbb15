import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def estimateE(Ig, Jg, Jgdx, Jgdy, x, y, window_size):

    left = math.floor(x-window_size[1]/2)
    top = math.floor(y-window_size[0]/2)
    right = left + window_size[1]
    bot = top + window_size[0]

    IgWin = Ig[top:bot,left:right]
    #plt.figure()
    #plt.imshow(IgWin, cmap='gray')
    JgWin = Jg[top:bot,left:right]
    #plt.figure()
    #plt.imshow(JgWin, cmap='gray')

    JgdxWin = Jgdx[top:bot,left:right]
    JgdyWin = Jgdy[top:bot,left:right]

    #plt.figure()
    #plt.imshow(JgdxWin, cmap='gray')
    #plt.figure()
    #plt.imshow(JgdyWin, cmap='gray')

    #plt.show()

    diff = IgWin-JgWin
    error = np.transpose([(np.multiply(diff,JgdxWin)).sum(), (np.multiply(diff,JgdyWin)).sum()])
    #error = error.sum(axis=1)
    #error = error.sum(axis=0)
    return error
