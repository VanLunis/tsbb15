import cvl_labs.lab4 as lab
import cv2
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def lpMake(ksize,sigma):
    lp = np.atleast_2d(np.exp(-0.5 * np.square(np.arange(-ksize,ksize+1,1)/sigma)))
    lp = lp/np.sum(lp)
    return lp

# def imgHessian(dx,dy,ksize,sigma,noLP=False):
#     if noLP:
#         df = np.atleast_2d(np.arange(-ksize,ksize+1,1))
#         df = df/np.abs(df).sum()
#         lp = np.atleast_2d([0,1,0])
#     else:
#         lp = lpMake(ksize,sigma)
#         df = np.atleast_2d(-1.0/np.square(sigma) * np.arange(-ksize,ksize+1,1) * lp)
#     H = np.zeros((dx.shape[0],dx.shape[1],2,2))
#     H[:,:,0,0] = conv2(conv2(dx,df,mode='same'), lp.T, mode='same')
#     H[:,:,0,1] = conv2(conv2(dx,lp,mode='same'), df.T, mode='same')
#     H[:,:,1,0] = H[:,:,0,1]
#     H[:,:,1,1] = conv2(conv2(dy,lp,mode='same'), df.T, mode='same')
#     return H

def imHessian(L):
    HL = np.zeros(L.shape + (3,))
    H11 = np.array([[1,-2,1],[6,-12,6],[1,-2,1]])/8.0 #papper
    # H11 = np.array([[0,0,0],[1,-2,1],[0,0,0]])/1.0 #F14
    H12 = np.array([[1,0,-1],[0,0,0],[-1,0,1]])/4.0
    H22 = H11.T
    HL[:,:,0] = conv2(L,H11,mode='same')
    HL[:,:,1] = conv2(L,H12,mode='same')
    HL[:,:,2] = conv2(L,H22,mode='same')
    return HL
# def imGrad(im,ksize,sigma):
#     lp = lpMake(ksize,sigma)
#     df = np.atleast_2d(-1.0/np.square(sigma) * np.arange(-ksize,ksize+1,1) * lp)
#     dx = conv2(conv2(im,df,mode='same'), lp.T, mode='same')
#     dy = conv2(conv2(im,lp,mode='same'), df.T, mode='same')
#     return dx,dy

def imGrad(im):
    dfilt = np.atleast_2d([-1,0,1])
    lp = np.atleast_2d([3,10,3])/32.0
    dxkern = lp.T@dfilt
    dykern = dfilt.T@lp
    dx = conv2(im,dxkern,mode='same')
    dy = conv2(im,dykern,mode='same')
    return dx,dy

def estimateT(dx,dy,windowSize=(3,3),mode='ones',ksize=2,sigma=2.0):
    T = np.zeros((L.shape[0],L.shape[1],3))
    if mode=='ones':
        window = np.ones(windowSize).astype('float64')
        T[:,:,0] = conv2(dx**2,window,mode='same')
        T[:,:,1] = conv2(dx*dy,window,mode='same')
        T[:,:,2] = conv2(dy**2,window,mode='same')
    elif mode=='gauss':
        lp = lpMake(ksize,sigma)
        T[:,:,0] = conv2(conv2(dx**2,lp,mode='same'),lp.T,mode='same')
        T[:,:,1] = conv2(conv2(dx*dy,lp,mode='same'),lp.T,mode='same')
        T[:,:,2] = conv2(conv2(dy**2,lp,mode='same'),lp.T,mode='same')
    return T

def estimateD(T,k):
    Tp = np.transpose(np.array([[T[:,:,0],T[:,:,1]],[T[:,:,1],T[:,:,2]]]),(2,3,0,1))
    W,V = np.linalg.eig(Tp)
    W = np.exp(-W/k)
    D = V@(W[:,:,:,None]*np.transpose(V,(0,1,3,2)))
    return np.transpose(np.array([D[:,:,0,0],D[:,:,0,1],D[:,:,1,1]]),(1,2,0))

size = 1000

# wgn = np.random.randn(size,size).astype('float64')
# lp = lpMake(size/2,size/8.0).astype('float64')
# lpwgn = conv2(conv2(wgn,lp,mode='same'), lp.T, mode='same')

# plt.figure(1)
# plt.imshow(lpwgn,interpolation='none')
# plt.show()

ksizeT = 2
sigmaT = 1.0
ksizeG = 2
sigmaG = 1.0
ksizeH = 5
sigmaH = 1.0
windowSize = (5,5)
k = 0.005
ds = 0.05
winW = 1000
winH = 1000

# http://liu.diva-portal.org/smash/get/diva2:265740/FULLTEXT01.pdf

L = lab.get_cameraman()/255.0
noise = np.random.randn(L.shape[0],L.shape[1])*0.05
# noise = (0.5-np.random.rand(L.shape[0],L.shape[1]))*0.2
L += noise
# L = mpimg.imread('cornertest.png')[:,:,0]
cv2.namedWindow('D',cv2.WINDOW_NORMAL) #cv2 suger dase
cv2.resizeWindow('D', 600, 600)
cv2.namedWindow('L',cv2.WINDOW_NORMAL)
cv2.resizeWindow('L', 600, 600)
cv2.namedWindow('H',cv2.WINDOW_NORMAL)
cv2.resizeWindow('H', 600, 600)
Lorig = np.array(L)
for i in range(1000):
    dx,dy = imGrad(L)
    T = estimateT(dx,dy,ksize=ksizeT,sigma=sigmaT,mode='gauss')
    H = imHessian(L)
    # H[:,:,1] = 0
    # D = np.exp(-T/k)
    D = estimateD(T,k)
    L += ds*((D*H).sum(2)+D[:,:,1]*H[:,:,1])
    cv2.imshow('D',D[:,:,0])
    cv2.imshow('L',L)
    cv2.imshow('H',np.abs(H[:,:,0]))
    cv2.waitKey()

# plt.figure(1)
# plt.imshow(Lorig)
# plt.figure(2)
# plt.imshow(L)
# plt.figure(3)
# plt.imshow(D[:,:,0,0])
# plt.show()
