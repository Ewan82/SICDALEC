import numpy as np
import SIC as SIC

def PlotsuccSIC(d,i,j,obs):
    Xlist=np.arange(i,j,1)
    SIClist=[-9999]*(j-i)
    for x in range(i,j):
         SIClist[x-i]=SIC.SIC(d,i,x+1,obs)

    import matplotlib.pyplot as plt

    plt.plot(Xlist,SIClist)
    plt.xlabel('Day of observation of NEE')
    plt.ylabel('Shannon Information Content')
    plt.title('SIC varying with successive observations of NEE')
    plt.show()
    
def PlotoneSIC(d,i,j,obs):
    Xlist=np.arange(i,j,1)
    SIClist=[-9999]*(j-i)
    for x in range(i,j):
        SIClist[x-i]=SIC.SIC(d,x,x+1,obs)
        
    #y=[sum(SIClist[:i+1]) for i in range(len(SIClist))]
    
    import matplotlib.pyplot as plt

    plt.plot(Xlist,SIClist)
    plt.xlabel('Day of observation of NEE')
    plt.ylabel('Shannon Information Content')
    plt.title('SIC for a single observation of NEE')
    plt.show()
