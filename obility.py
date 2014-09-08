import numpy as np
import Model as Mod

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
    
def Obility(d,i,j):
    H=Hmat(d,i,j)
    
    return np.linalg.matrix_rank(H)
