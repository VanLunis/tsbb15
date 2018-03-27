import numpy as np
import scipy
from scipy.signal import convolve2d as conv2
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from toGray import rgb2gray
#image = misc.ascent()

image = rgb2gray(mpimg.imread('baboon.png'))

lp = np.ones((1,9), dtype='float32')/9.0
p2fs = conv2(lp, np.transpose(lp))

print(p2fs.shape)
print(lp.shape)
print('BILDEN: ')
print(image.shape)

newIm = conv2(image,p2fs,'same')

plt.figure()
plt.imshow(newIm)

plt.show()
