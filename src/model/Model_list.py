import Model as Mod
import numpy as np
import ad

def Modlist(d,i,j):
    Mlist=[-9999]*(j-i)
    Clist=d.Clist[:]+[-9999]*(j-i)
    for x in xrange(i,j):    
        Cf=Clist[x-i][0]
        Cr=Clist[x-i][1]
        Cw=Clist[x-i][2]
        Cl=Clist[x-i][3]
        Cs=Clist[x-i][4]
        Mlist[x-i]=Mod.LinDALEC(Cf,Cr,Cw,Cl,Cs,x,d)  
        Clist[(x+1)-i]=Mod.DALEC(Cf,Cr,Cw,Cl,Cs,x,d)
    
    return Clist,Mlist
