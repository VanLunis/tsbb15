import cvl_labs.lab4 as lab
import cv2
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def imHessian(L):
    HL = np.zeros(L.shape + (3,))
    # H11 = np.array([[1,-2,1],[6,-12,6],[1,-2,1]])/8.0 #papper
    H11 = np.array([[0,0,0],[1,-2,1],[0,0,0]])/1.0 #F14
    H12 = np.array([[1,0,-1],[0,0,0],[-1,0,1]])/4.0
    H22 = np.array(H11.T)
    HL[:,:,0] = conv2(L,H11,mode='same')
    HL[:,:,1] = conv2(L,H12,mode='same')
    HL[:,:,2] = conv2(L,H22,mode='same')
    return HL

def imGrad(im):
    dfilt = np.atleast_2d([-1,0,1])
    lp = np.atleast_2d([3,10,3])/32.0
    dxkern = np.array(lp.T@dfilt)
    dykern = np.array(dfilt.T@lp)
    dx = conv2(im,dxkern,mode='same')
    dy = conv2(im,dykern,mode='same')
    return dx,dy


alpha = 0.1
lam = 0.01

L = lab.get_cameraman()/255.0

u = np.random.rand(512,512)
X = np.zeros(u.shape)
X[::2,::2] = 1

u[::2,::2] = np.array(L)
g = np.array(u)

window1 = cv2.namedWindow('u',cv2.WINDOW_NORMAL) #cv2 suger dase
cv2.resizeWindow('u', 600, 600)
window1 = cv2.namedWindow('dx',cv2.WINDOW_NORMAL) #cv2 suger dase
cv2.resizeWindow('dx', 600, 600)
window1 = cv2.namedWindow('dy',cv2.WINDOW_NORMAL) #cv2 suger dase
cv2.resizeWindow('dy', 600, 600)


for i in range(1000):
    dx,dy = imGrad(u)
    H = imHessian(u)
    dxx = H[:,:,0]
    dxy = H[:,:,1]
    dyy = H[:,:,2]
    gradterm = np.sqrt(dx**2+dy**2)**3
    gradterm[gradterm < 0.000001] = 0.000001
    u += - alpha*(X*(u-g) - lam*(dxx*(dy**2)-2*dxy*dx*dy+dyy*(dx**2))/gradterm)
    cv2.imshow('u',u)
    cv2.imshow('dx',dx)
    cv2.imshow('dy',dy)
    cv2.waitKey()
