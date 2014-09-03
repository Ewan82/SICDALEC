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

def Hmat(i,j):
    Cf=58.0
    Mlist=[-9999]*(j-i)
    Hlist=[-9999]*(j-i)

    for x in range(i,j):
        GPPdiff=Mod.GPPdiff(Cf,float(d.phi_d[x]),float(d.T_range[x]),float(d.R_tot[x]),float(d.T_max[x]),float(d.D[x]),float(d.I[x]))
        Hlist[x-i]=np.matrix([[-(1-d.p_2)*GPPdiff, 0, 0, d.p_8*d.T[x], d.p_9*d.T[x]]])
        Mlist[x-i]=np.matrix([[(1-d.p_5)+d.p_3*(1-d.p_2)*GPPdiff,0,0,0,0],[d.p_4*(1-d.p_3)*(1-d.p_2)*GPPdiff,(1-d.p_7),0,0,0],[(1-d.p_4)*(1-d.p_3)*(1-d.p_2)*GPPdiff,0,(1-d.p_6),0,0],[d.p_5,d.p_7,0,1-(d.p_1+d.p_8)*d.T[x],0],[0,0,d.p_6,d.p_1*d.T[x],1-d.p_9*d.T[x]]])
	
    stacklist=[Hlist[0]]+[-9999]*(j-i-1) #Creates H hat matrix
    for x in range(1,j-i):
        stacklist[x]=Hlist[x]*Mfac(Mlist,x-1)

    Hmat=np.vstack(stacklist)
    
    return Hmat


def SIC(i,j):

    H=Hmat(i,j)

    sigO_NEE=0.25
    R=np.matrix(sigO_NEE*np.identity(j-i))	
	
    B=np.matrix([[d.sigB_cf,0,0,0,0],[0,d.sigB_cw,0,0,0],[0,0,d.sigB_cr,0,0],[0,0,0,d.sigB_cl,0],[0,0,0,0,d.sigB_cs]]) #Background error covariance matrix

    J2nddiff=B.I+H.T*R.I*H #Calculates Hessian

    SIC=0.5*np.log((np.linalg.det(J2nddiff))*(np.linalg.det(B))) #Calculates SIC

    return SIC	

	
