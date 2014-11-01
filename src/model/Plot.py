import numpy as np
import SIC as SIC
import Model as Mod
import copy as cp
import matplotlib.pyplot as plt


def PlotsuccSIC(d,i,j,obs):
    Mlist = Mod.Mlist(d,i,j)
    Xlist = np.arange(i,j,1)
    SIClist = np.ones(j-i)*-9999
    
    for x in xrange(i,j):
         SIClist[x-i]=SIC.SIC(d,i,x+1,obs,Mlist)

    plt.plot(Xlist, SIClist)
    plt.xlabel('Day of observation', fontsize=20)
    plt.ylabel('Shannon Information Content', fontsize=20)
    plt.title('SIC varying with successive observations', fontsize=20)
    plt.show()
 
   
def PlotoneSIC(d,i,j,obs):
    Mlist = Mod.Mlist(d,i,j)
    Xlist = np.arange(i,j,1)
    SIClist = np.ones(j-i)*-9999.
    
    for x in xrange(i,j):
        SIClist[x-i] = SIC.SIC(d, x, x+1, obs, Mlist)

    plt.plot(Xlist,SIClist)
    plt.xlabel('Day of observation of NEE', fontsize=20)
    plt.ylabel('Shannon Information Content', fontsize=20)
    plt.title('SIC for a single observation of NEE', fontsize=20)
    plt.show()


def Plottemp(d,i,j):
    Xlist = np.arange(i,j,1)
    Tlist = d.T[i:j]

    plt.plot(Xlist, Tlist)
    plt.xlabel('Day of observation of Temperature', fontsize=20)
    plt.ylabel(r'Temperature term, $\frac{1}{2}exp(\Theta T_{mean})$',\
               fontsize=20)
    plt.title('Temperature term for three years of data', fontsize=20)
    plt.show()
    
    
def PlotsuccDOFS(d, i, j, obs):
    Mlist = Mod.Mlist(d, i, j)
    Xlist = np.arange(i, j, 1)
    DOFSlist = np.ones(j-i)*-9999.
    
    for x in xrange(i,j):
         DOFSlist[x-i] = SIC.DOFS(d,i,x+1,obs,Mlist)

    plt.plot(Xlist,DOFSlist, label=obs)
    plt.legend(loc='lower right')
    #plt.xlabel('Day of observation', fontsize=20)
    plt.ylabel('DFS', fontsize=25)
    #plt.title('DFS varying with successive observations', fontsize=20)
    
    
def PlotoneDOFS(d, i, j, obs):
    Mlist = Mod.Mlist(d, i, j)
    Xlist = np.arange(i, j, 1)
    DOFSlist = np.ones(j-i)*-9999.
    
    for x in xrange(i, j):
        DOFSlist[x-i] = SIC.DOFS(d,x,x+1,obs,Mlist)

    plt.plot(Xlist, DOFSlist)
    plt.xlabel('Day of observation', fontsize=20)
    plt.ylabel('Degrees of freedom for signal', fontsize=20)
    plt.title('DFS for a single observation of NEE', fontsize=20)
    plt.show()

    
def PlotLinModerr(d, i, j, Cpool):
    pooldict={'cf':0, 'cr':1, 'cw':2, 'cl':3, 'cs':4}
    
    Xlist = np.arange(i, j+1, 1)

    Clistx = Mod.Clist(d, i, j)

    d2 = cp.copy(d)
    d2.Clist = d2.Clist*1.05
    Clistxdx = Mod.Clist(d2, i, j)

    d3 = cp.copy(d)
    d3.Clist = d3.Clist*0.05
    Matlist = Mod.Mlist(d, i, j)
    dxL = Mod.Clist_lin(d3, i, j, Matlist)

    #dxN = Clistxdx - Clistx
    
    err = abs(((Clistxdx - Clistx)/dxL)-1)*100

    plt.plot(Xlist, err[:,pooldict[Cpool]], label=Cpool)
    #plt.plot(Xlist, dxN[:,pooldict[Cpool]], label='dxN '+Cpool)
    #plt.plot(Xlist, dxL[:,pooldict[Cpool]], label='dxL '+Cpool)
    plt.legend(loc='upper left')
    #plt.show()
    
    
def Subplot_err(d, i, j):
    plt.subplot(3,2,1)   
    PlotLinModerr(d, i, j, 'cf')
    plt.ylabel('Error (%)', fontsize=20)
    #plt.title('Plot of dxN and dxL')
    #plt.legend()
    plt.subplot(3,2,2) 
    PlotLinModerr(d, i, j, 'cr')
    #plt.title('Plot of dxN and dxL')
    #plt.legend()
    plt.subplot(3,2,3) 
    PlotLinModerr(d, i, j, 'cs')
    plt.ylabel('Error (%)', fontsize=20)
    #plt.legend()
    plt.subplot(3,2,4)
    PlotLinModerr(d, i, j, 'cl')
    #plt.legend()
    plt.subplot(3,1,3)
    PlotLinModerr(d, i, j, 'cw')
    plt.xlabel('Day of model run',fontsize=20)
    plt.ylabel('Error (%)',fontsize=20)
    plt.suptitle(r"Plot of $ (\frac{m( \mathbf{x} + \delta \mathbf{x})"
                 r"-m(\mathbf{x})}{\mathbf{M}\delta\mathbf{x}}-1) \times"
                 r"100$ for each carbon pool", fontsize=20) 
    #plt.legend(loc='lower right')
    #plt.suptitle(r'Plot of $\bigg(\frac{m(\mathbf{x}+\delta\mathbf{x})-m(\mathbf{x})}{\mathbf{M}\delta\mathbf{x}}-1\bigg)\times 100$ for each carbon pool', fontsize=20)    
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
    plt.xlabel('Day of observation', fontsize=25)
    plt.subplot(3,2,6)
    PlotsuccDOFS(d, i, j, 'nee')
    plt.xlabel('Day of observation', fontsize=25)
    plt.suptitle('DFS varying with successive observations', fontsize=25)
    plt.show()

    
def PlotsuccSIClist(d,i,j,obs):
    Mlist = Mod.Mlist(d, i, j)
    SIClist = np.ones(j-i)*-9999.
    
    for x in xrange(i,j):
         SIClist[x-i] = SIC.SIC(d, i, x+1, obs, Mlist)
    return SIClist
    
    
def PlotsuccDOFSlist(d,i,j,obs):
    Mlist = Mod.Mlist(d, i, j)
    DOFSlist = np.ones(j-i)*-9999.
    
    for x in xrange(i, j):
         DOFSlist[x-i] = SIC.DOFS(d, i, x+1, obs, Mlist)
    return DOFSlist
    
    
def Plotwintsumm(d, i, j, obs):
    Mlist = Mod.Mlist(d, i , j)
    xlist = np.arange(0, j-i, 1)
    sumsiclist = np.ones(j-i)*SIC.SIC(d, 923, 924, obs, Mlist)
    wint = PlotsuccSIClist(d, i, j, obs)
    plt.plot(xlist, wint, label='successive winter obs')    
    plt.plot(xlist, sumsiclist, '--', label='single summer obs')
    plt.xlabel('Day', fontsize=20)
    plt.ylabel('Shannon Information Content', fontsize=20)
    plt.title('SIC for NEE winter and summer', fontsize=20)
    plt.legend(loc='upper left')
    
    
def PlotwintsummDOFs(d,i,j,obs):
    Mlist = Mod.Mlist(d, i , j)
    xlist = np.arange(0, j-i, 1)
    sumsiclist = np.ones(j-i)*SIC.DOFS(d, 923, 924, obs, Mlist)
    wint = PlotsuccDOFSlist(d, i, j, obs)
    plt.plot(xlist, wint, label='successive winter obs')    
    plt.plot(xlist, sumsiclist, '--', label='single summer obs')
    plt.xlabel('Day', fontsize=20)
    plt.ylabel('Degrees of Freedom for Signal', fontsize=20)
    plt.title('DFS for NEE winter and summer', fontsize=20)
    plt.legend(loc='upper left')