import cvl_labs.lab3 as lab
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

# corrLists are assumed to contain putative/tentative correspondences
def ransacAlg(corrList1, corrList2, inlierThresh, maxIter):
    bestInliers = 0
    bestF = 0
    numPoints = corrList1.shape[1]

    for iteration in range(maxIter):
        # print(iteration)
        pointList = []
        firstPoints = np.zeros((2,8))
        secondPoints = np.zeros((2,8))

        #Extracts 8 random corresponding points
        while (len(pointList) < 8):
            point = rnd.randint(0, numPoints-1)
            if (point not in pointList):
                pointList.append(point)
                firstPoints[:,len(pointList)-1] = corrList1[:,point]
                secondPoints[:,len(pointList)-1] = corrList2[:,point]

        F = lab.fmatrix_stls(firstPoints,secondPoints)
        resid = lab.fmatrix_residuals(F, corrList1, corrList2)
        inliers = ((np.abs(resid) < inlierThresh).sum(0) == 2)
        numInliers = inliers.sum()


        if (numInliers > bestInliers):
            bestF = np.array(F)
            bestInliers = int(numInliers)
            print(bestInliers)
            bestCorr1 = np.array([corrList1[0][inliers],corrList1[1][inliers]])
            bestCorr2 = np.array([corrList2[0][inliers],corrList2[1][inliers]])

    return bestF, bestCorr1, bestCorr2
