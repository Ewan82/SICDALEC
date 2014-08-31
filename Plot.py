import numpy as np
import SIC as SIC

def Plot(i,j):
    Xlist=np.arange(i,j,1)
    SIClist=[]
    for x in range(i,j):
         SIClist.append(SIC.SIC(i,x+1))

    import matplotlib.pyplot as plt

    plt.plot(Xlist,SIClist)
    plt.xlabel('Day of observation of NEE')
    plt.ylabel('Shannon Information Content')
    plt.title('SIC varying with successive observations of NEE')
    plt.show()