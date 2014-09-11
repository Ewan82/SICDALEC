#This will have code in
import numpy as np
import Model as Mod
import sys
import ad

def Mfac(Mlist,a): #Matrix factoral to find product of M matrices
    if a==0:
        return Mlist[0]
    else:
        Mat=Mlist[0]
        for x in xrange(0,a):
            Mat=Mat*Mlist[x+1]
        return Mat

def Hmat(d,i,j,obs): #Creates H^hat matrix for given obs, i,j and data d
  
    Hlist=[-9999]*(j-i) 
    
    Clist,Mlist=Mod.Modlist(d,i,j)
    
    obsdict={'NEE': Mod.NEE, 'LF': Mod.LF, 'LW': Mod.LW, 'Cf': Mod.Cf, 'Cr': Mod.Cr, 'Cw': Mod.Cw, 'Cl': Mod.Cl, 'Cs': Mod.Cs}

    for x in range(i,j):
        Cf=ad.adnumber(Clist[x-i][0])
        Cr=ad.adnumber(Clist[x-i][1])
        Cw=ad.adnumber(Clist[x-i][2])
        Cl=ad.adnumber(Clist[x-i][3])
        Cs=ad.adnumber(Clist[x-i][4])
        Hlist[x-i]=ad.jacobian(obsdict[obs](Cf,Cr,Cw,Cl,Cs,x,d),[Cf,Cr,Cw,Cl,Cs])
	
    stacklist=[Hlist[0]]+[-9999]*(j-i-1) #Creates H hat matrix
    for x in range(1,j-i):
        stacklist[x]=Hlist[x]*Mfac(Mlist,x-1)

    Hmat=np.vstack(stacklist)
    
    return np.matrix(Hmat)
    


def SIC(d,i,j,obs): #Calculates value of Shannon Info Content (SIC=0.5*ln(|B|/|A|), measure of reduction in entropy given a set of observations) 
    
    stdO_dict={'NEE': d.sigO_nee, 'LF': d.sigO_lf, 'LW': d.sigO_lw, 'Cf': d.sigO_cf, 'Cr': d.sigO_cr, 'Cw': d.sigO_cw, 'Cl': d.sigO_cl, 'Cs': d.sigO_cs}
    
    H=Hmat(d,i,j,obs)

    sigO=stdO_dict[obs]
    R=np.matrix(sigO*np.identity(j-i))	
	
    B=d.B #Background error covariance matrix

    J2nddiff=B.I+H.T*R.I*H #Calculates Hessian

    SIC=0.5*np.log((np.linalg.det(J2nddiff))*(np.linalg.det(B))) #Calculates SIC

    return SIC	

	
