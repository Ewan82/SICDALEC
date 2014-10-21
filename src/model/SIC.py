#This will have code in
import numpy as np
import Model as Mod
import ad
import re


def Mfac(Mlist,a): #Matrix factoral to find product of M matrices
    if a==0:
        return np.matrix(Mlist[0])
    else:
        Mat=np.matrix(Mlist[0])
        for x in xrange(0,a):
            Mat=np.matrix(Mlist[x+1])*Mat
        return Mat


def Hmat(d,i,j,obs_str,Mlist): #Creates H^hat matrix for given obs, i,j and data d
  
    Hlist=[-9999]*(j-i) 
    
    Obslist=re.findall(r'[^,;\s]+', obs_str)
    
    Hhold=[-9999]*len(Obslist)
    
    obsdict={'nee': Mod.NEE, 'lf': Mod.LF, 'lw': Mod.LW, 'cf': Mod.Cf, \
             'cr': Mod.Cr, 'cw': Mod.Cw, 'cl': Mod.Cl, 'cs': Mod.Cs}

    for x in xrange(i,j):
        Cf=ad.adnumber(d.Cf) #Clist[x-i][0]) #Redo this!!!
        Cr=ad.adnumber(d.Cr) #Clist[x-i][1])
        Cw=ad.adnumber(d.Cw) #Clist[x-i][2])
        Cl=ad.adnumber(d.Cl) #Clist[x-i][3])
        Cs=ad.adnumber(d.Cs) #Clist[x-i][4])
        for y in range(0,len(Obslist)):
            Hhold[y]=ad.jacobian(obsdict[Obslist[y]](Cf,Cr,Cw,Cl,Cs,x,d),[Cf,Cr,Cw,Cl,Cs])
        Hlist[x-i]=np.vstack(Hhold)
	
    stacklist=[Hlist[0]]+[-9999]*(j-i-1) #Creates H hat matrix
    for x in xrange(1,j-i):
        stacklist[x]=Hlist[x]*Mfac(Mlist,x-1)

    Hmat=np.vstack(stacklist)
    
    return np.matrix(Hmat)
    
    
def Rmat(d,i,j,obs_str):

    stdO_dict={'nee': d.sigO_nee, 'lf': d.sigO_lf, 'lw': d.sigO_lw, \
               'cf': d.sigO_cf, 'cr': d.sigO_cr, 'cw': d.sigO_cw, \
               'cl': d.sigO_cl, 'cs': d.sigO_cs}

    Obslist=re.findall(r'[^,;\s]+', obs_str)
    
    sigO=[-9999]*len(Obslist)

    for y in xrange(0,len(Obslist)):
        sigO[y]=stdO_dict[Obslist[y]]
        
    R=np.matrix(np.identity((j-i)*len(Obslist)))
    sigR=sigO*(j-i)
    for x in xrange(0,(j-i)*len(sigO)):
        R[x,x]=sigR[x]
            
    return R
      
    
def SIC(d,i,j,obs_str,Mlist): #Calculates value of Shannon Info Content (SIC=0.5*ln(|B|/|A|), measure of reduction in entropy given a set of observations)
    
    H=Hmat(d,i,j,obs_str,Mlist)
    
    R=Rmat(d,i,j,obs_str)
    	
    B=d.B #Background error covariance matrix

    J2nddiff=B.I+H.T*R.I*H #Calculates Hessian

    SIC=0.5*np.log((np.linalg.det(J2nddiff))*(np.linalg.det(B))) #Calculates SIC

    return SIC	


def DOFS(d,i,j,obs,Mlist):
    H=Hmat(d,i,j,obs,Mlist)
    R=Rmat(d,i,j,obs)
    B=d.B
    A=(B.I+H.T*R.I*H).I
    DOFS=len(d.Clist[0])-np.trace(B.I*A)
    return DOFS
	
	
def Obility(d,i,j,obs):
    H=Hmat(d,i,j,obs)
    
    return np.linalg.matrix_rank(H)
