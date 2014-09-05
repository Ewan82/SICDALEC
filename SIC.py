#This will have code in
import numpy as np
import Model as Mod
import Data as D
d=D.dalecData()

def Mfac(Mlist,a): #Matrix factoral to find product of M matrices
    if a==0:
        return Mlist[0]
    else:
        Mat=Mlist[0]
        for x in xrange(0,a):
            Mat=Mat*Mlist[x+1]
        return Mat

def Hmat(d,i,j):
    Cf=58.0
    Hlist=[-9999]*(j-i)
    Mlist=Mod.Modlist(d,i,j)[1]

    for x in range(i,j):
        GPPdiff=Mod.GPPdiff(Cf,d,x)
        Hlist[x-i]=np.matrix([[-(1-d.p_2)*GPPdiff, 0, 0, d.p_8*d.T[x], d.p_9*d.T[x]]])
	
    stacklist=[Hlist[0]]+[-9999]*(j-i-1) #Creates H hat matrix
    for x in range(1,j-i):
        stacklist[x]=Hlist[x]*Mfac(Mlist,x-1)

    Hmat=np.vstack(stacklist)
    
    return Hmat


def SIC(d,i,j):

    H=Hmat(d,i,j)

    sigO_NEE=0.25
    R=np.matrix(sigO_NEE*np.identity(j-i))	
	
    B=np.matrix([[d.sigB_cf,0,0,0,0],[0,d.sigB_cw,0,0,0],[0,0,d.sigB_cr,0,0],[0,0,0,d.sigB_cl,0],[0,0,0,0,d.sigB_cs]]) #Background error covariance matrix

    J2nddiff=B.I+H.T*R.I*H #Calculates Hessian

    SIC=0.5*np.log((np.linalg.det(J2nddiff))*(np.linalg.det(B))) #Calculates SIC

    return SIC	

	
