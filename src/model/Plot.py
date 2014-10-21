import numpy as np
import SIC as SIC
import Model as Mod
import copy as cp
import matplotlib.pyplot as plt

def PlotsuccSIC(d,i,j,obs):
    Mlist=Mod.Mlist(d,i,j)
    Xlist=np.arange(i,j,1)
    SIClist=np.ones(j-i)*-9999
    for x in xrange(i,j):
         SIClist[x-i]=SIC.SIC(d,i,x+1,obs,Mlist)

    plt.plot(Xlist,SIClist)
    plt.xlabel('Day of observation')
    plt.ylabel('Shannon Information Content')
    plt.title('SIC varying with successive observations')
    plt.show()
 
   
def PlotoneSIC(d,i,j,obs):
    Mlist=Mod.Mlist(d,i,j)
    Xlist=np.arange(i,j,1)
    SIClist=[-9999]*(j-i)
    for x in xrange(i,j):
        SIClist[x-i]=SIC.SIC(d,x,x+1,obs,Mlist)
        
    #y=[sum(SIClist[:i+1]) for i in range(len(SIClist))]

    plt.plot(Xlist,SIClist)
    plt.xlabel('Day of observation of NEE', fontsize=20)
    plt.ylabel('Shannon Information Content', fontsize=20)
    plt.title('SIC for a single observation of NEE', fontsize=20)
    plt.show()


def Plottemp(d,i,j):
    Xlist=np.arange(i,j,1)
    Tlist = d.T[i:j]
    #y=[sum(SIClist[:i+1]) for i in range(len(SIClist))]

    plt.plot(Xlist,Tlist)
    plt.xlabel('Day of observation of Temperature', fontsize=20)
    plt.ylabel(r'Temperature term, $\frac{1}{2}exp(\Theta T_{mean})$',\
               fontsize=20)
    plt.title('Temperature term for three years of data', fontsize=20)
    plt.show()
    
    
def PlotsuccDOFS(d,i,j,obs):
    Mlist=Mod.Mlist(d,i,j)
    Xlist=np.arange(i,j,1)
    DOFSlist=[-9999]*(j-i)
    for x in xrange(i,j):
         DOFSlist[x-i]=SIC.DOFS(d,i,x+1,obs,Mlist)

    plt.plot(Xlist,DOFSlist)
    #plt.xlabel('Day of observation')
    plt.ylabel('DOFS')
    plt.title('DOFS varying with successive observations of '+obs)

    
def PlotLinModerr(d, i, j, Cpool):
    pooldict={'cf':0, 'cr':1, 'cw':2, 'cl':3, 'cs':4}
    
    Xlist = np.arange(i, j+1, 1)

    Clistx = Mod.Clist(d, i, j)

    d2 = cp.copy(d)
    d2.Clist = d2.Clist*1.1
    Clistxdx = Mod.Clist(d2, i, j)

    d3 = cp.copy(d)
    d3.Clist = d3.Clist*0.1
    dxL = Mod.Clist_lin(d3, i, j)

    dxN = Clistxdx - Clistx

    plt.plot(Xlist, dxN[:,pooldict[Cpool]], label='dxN '+Cpool)
    plt.plot(Xlist, dxL[:,pooldict[Cpool]], label='dxL '+Cpool)
    plt.legend()
    #plt.show()
    
    
def Subplot_err(d, i, j):
    plt.subplot(3,2,1)
    PlotLinModerr(d, i, j, 'cf')
    plt.ylabel('Carbon pool value')
    plt.title('Plot of dxN and dxL')
    plt.subplot(3,2,2) 
    PlotLinModerr(d, i, j, 'cr')
    plt.title('Plot of dxN and dxL')
    plt.subplot(3,2,3) 
    PlotLinModerr(d, i, j, 'cs')
    plt.xlabel('Day of model run')
    plt.ylabel('Carbon pool value') 
    plt.subplot(3,2,4)
    PlotLinModerr(d, i, j, 'cl')
    plt.xlabel('Day of model run')
    plt.subplot(3,1,3)
    PlotLinModerr(d, i, j, 'cw')
    plt.xlabel('Day of model run')
    plt.ylabel('Carbon pool value')
    plt.title('Plot of dxN and dxL') 
    plt.show()


def Subplot_DOFS(d, i, j):
    plt.subplot(3,2,1)
    PlotsuccDOFS(d, i, j, 'cf')
    plt.subplot(3,2,2) 
    PlotsuccDOFS(d, i, j, 'cr')
    plt.subplot(3,2,3) 
    PlotsuccDOFS(d, i, j, 'cs')
    plt.subplot(3,2,4)
    PlotsuccDOFS(d, i, j, 'cl')
    plt.subplot(3,2,5)
    PlotsuccDOFS(d, i, j, 'cw')
    plt.xlabel('Day of observation')
    plt.subplot(3,2,6)
    PlotsuccDOFS(d, i, j, 'nee')
    plt.xlabel('Day of observation')
    plt.show()