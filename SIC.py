#This will have code in
import numpy as np
import Model as Mod
import sys
import ad
import re


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
    
    Obslist=re.findall(r'[^,;\s]+', obs)
    
    Hhold=[-9999]*len(Obslist)
    
    obsdict={'NEE': Mod.NEE, 'LF': Mod.LF, 'LW': Mod.LW, 'Cf': Mod.Cf, 'Cr': Mod.Cr, 'Cw': Mod.Cw, 'Cl': Mod.Cl, 'Cs': Mod.Cs}

    for x in range(i,j):
        Cf=ad.adnumber(d.Cf) #Clist[x-i][0])
        Cr=ad.adnumber(d.Cr) #Clist[x-i][1])
        Cw=ad.adnumber(d.Cw) #Clist[x-i][2])
        Cl=ad.adnumber(d.Cl) #Clist[x-i][3])
        Cs=ad.adnumber(d.Cs) #Clist[x-i][4])
        for y in range(0,len(Obslist)):
            Hhold[y]=ad.jacobian(obsdict[Obslist[y]](Cf,Cr,Cw,Cl,Cs,x,d),[Cf,Cr,Cw,Cl,Cs])
        Hlist[x-i]=np.vstack(Hhold)
	
    stacklist=[Hlist[0]]+[-9999]*(j-i-1) #Creates H hat matrix
    for x in range(1,j-i):
        stacklist[x]=Hlist[x]*Mfac(Mlist,x-1)

    Hmat=np.vstack(stacklist)
    
    return np.matrix(Hmat)
    
    
def Rmat(d,i,j,obs):

    stdO_dict={'NEE': d.sigO_nee, 'LF': d.sigO_lf, 'LW': d.sigO_lw, 'Cf': d.sigO_cf, 'Cr': d.sigO_cr, 'Cw': d.sigO_cw, 'Cl': d.sigO_cl, 'Cs': d.sigO_cs}

    Obslist=re.findall(r'[^,;\s]+', obs)
    
    sigO=[-9999]*len(Obslist)

    for y in range(0,len(Obslist)):
        sigO[y]=stdO_dict[Obslist[y]]
        
    R=np.matrix(np.identity((j-i)*len(Obslist)))
    sigR=sigO*(j-i)
    for x in range(0,(j-i)*len(sigO)):
        R[x,x]=sigR[x]
            
    return R
      
    
def SIC(d,i,j,obs): #Calculates value of Shannon Info Content (SIC=0.5*ln(|B|/|A|), measure of reduction in entropy given a set of observations) 
    
    H=Hmat(d,i,j,obs)
    
    R=Rmat(d,i,j,obs)
    	
    B=d.B #Background error covariance matrix

    J2nddiff=B.I+H.T*R.I*H #Calculates Hessian

    SIC=0.5*np.log((np.linalg.det(J2nddiff))*(np.linalg.det(B))) #Calculates SIC

    return SIC	


def DOFS(d,i,j,obs):
    H=Hmat(d,i,j,obs)
    R=Rmat(d,i,j,obs)
    B=d.B
    A=(B.I+H.T*R.I*H).I
    DOFS=len(d.Clist[0])-np.trace(B.I*A)
    return DOFS
	
	
def Obility(d,i,j,obs):
    H=Hmat(d,i,j,obs)
    
    return np.linalg.matrix_rank(H)
