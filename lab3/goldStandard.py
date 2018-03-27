import cvl_labs.lab3 as lab
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from scipy.optimize import least_squares

# corrLists are assumed to contain putative/tentative correspondences
def goldStandAlg(corrList1, corrList2, initF=None):
    numPoints = corrList1.shape[1]
    # Creates initial estimation of F
    if initF is None:
        initF = lab.fmatrix_stls(corrList1,corrList2)

    # Initial estimation of cameras
    C1, C2 = lab.fmatrix_cameras(initF)

    # Initial estimation of 3D points
    X = np.zeros((3,0))
    for currPoint in range(numPoints):
        X = np.concatenate((X,np.array(lab.triangulate_optimal(C1,C2,np.array(corrList1)[:,currPoint],np.array(corrList2)[:,currPoint]))[:,None]),axis=1)
    print(X.shape)
    params = np.hstack((C1.ravel(),X.T.ravel()))
    resid = lab.fmatrix_residuals_gs(params,corrList1,corrList2)
    result = least_squares(lab.fmatrix_residuals_gs,params,args=(corrList1,corrList2))

    C1 = np.array(result.x[0:12].reshape((3,4)))
    resX = np.array(result.x[12:].reshape((numPoints,3)).T)

    Fdone = lab.fmatrix_from_cameras(C1,C2)
    return Fdone,C1,C2, resX
