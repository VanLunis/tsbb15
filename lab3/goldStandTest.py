import cvl_labs.lab3 as lab
import matplotlib.pyplot as plt
import numpy as np
from goldStandard import goldStandAlg
from ransac import ransacAlg

roisize = 15
block_size = 5
kernel_size = 5
supress_of_max = 0.01
supress_size = 7
thresh = 1000


img1,img2 = lab.load_stereo_pair()
# interest points for img1
H1 = lab.harris(img1,block_size,kernel_size)
H1sup = lab.non_max_suppression(H1,supress_size)
intP1 = np.nonzero(H1sup > H1sup.max()*supress_of_max)
# interest points for img2
H2 = lab.harris(img2,block_size,kernel_size)
H2sup = lab.non_max_suppression(H2,supress_size)
intP2 = np.nonzero(H2sup > H2sup.max()*supress_of_max)


# compute putative correspondences
roi1 = lab.cut_out_rois(img1,intP1[1],intP1[0],roisize)
roi2 = lab.cut_out_rois(img2,intP2[1],intP2[0],roisize)



diffMat = np.zeros((len(roi1),len(roi2)))
for index1 in range(len(roi1)):
    for index2 in range(len(roi2)):
        if (roi1[index1].shape == (roisize,roisize)) & (roi2[index2].shape == (roisize,roisize)):
            diffMat[index1,index2] = ((roi1[index1].astype(float)-roi2[index2].astype(float))**2).sum()
        else:
            diffMat[index1,index2] = np.inf

vals,ri,ci = lab.joint_min(diffMat)

corrList1 = np.zeros((2,0)) # de vill ha en (2,N) size array
corrList2 = np.zeros((2,0))

corrList1 = np.concatenate((corrList1,np.array([intP1[1][ri],intP1[0][ri]])),axis=1)
corrList2 = np.concatenate((corrList2,np.array([intP2[1][ci],intP2[0][ci]])),axis=1)

Fransac, rans1, rans2 = ransacAlg(corrList1,corrList2,1,30000)
Fgolden,C1,C2,Xgolden = goldStandAlg(rans1,rans2, Fransac)
p1 = np.zeros((2,0))
p2 = np.zeros((2,0))
for x in range(Xgolden.shape[1]):
    p1 = np.concatenate((p1,lab.project(Xgolden.T[x],C1)[:,None]),axis=1)
    p2 = np.concatenate((p2,lab.project(Xgolden.T[x],C2)[:,None]),axis=1)



lab.show_corresp(img1,img2,p1,p2)

goldenResult = np.sqrt(((lab.fmatrix_residuals(Fgolden,rans1,rans2))**2).sum())
print("Cost function value:"+str(goldenResult))

ransacResult = np.sqrt(((lab.fmatrix_residuals(Fransac,rans1,rans2))**2).sum())
print("Cost function value:"+str(ransacResult))

print("inlier ratio: " + str(len(rans1[0])/len(corrList1[0])))

plt.figure()
lab.imshow(img1)
lab.plot_eplines(Fgolden,rans2,img1.shape)
plt.figure()
lab.imshow(img2)
lab.plot_eplines(Fgolden.T,rans1,img2.shape)

plt.show()
