import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def imagebw(imgToShow,thresh):
    image = mpimg.imread(imgToShow)
    plt.figure()
    if (thresh == 0):
        imgplot = plt.imshow(image)
    else:
        plt.imshow(image, clim=(thresh/(np.amax(image)*255), np.amax(image)), cmap='gray')
    plt.colorbar();
    plt.show()
    return

imagebw("baboon.png",0)
