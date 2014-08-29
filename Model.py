#ACM model for GPP
import numpy as np
import Data as D
d=D.dalecData()


def ACM(Cf, phi_d, T_range, R_tot, T_max, D, I):
    L=Cf/111.
    q=d.a_3-d.a_4
    gc=((abs(phi_d))**(d.a_10))/(0.5*T_range+d.a_6*R_tot)
    p=((d.a_1*d.N*L)/gc)*np.exp(d.a_8*T_max)
    Ci=0.5*(d.C_a+q-p+np.sqrt((d.C_a+q-p)**2-4*(d.C_a*q-p*d.a_3)))
    E0=(d.a_7*L**2)/(L**2+d.a_9)
    delta=-0.408*np.cos(((360*(D+10)*np.pi)/(365*180)))
    s=24*np.arccos((-np.tan(d.bigdelta)*np.tan(delta)))/np.pi
    GPP=(E0*I*gc*(d.C_a-Ci)*(d.a_2*s+d.a_5))/(E0*I+gc*(d.C_a-Ci))
    
    return GPP
    
    
def SIC(i,j):

    GPPlist=[-999]*(j-i)
    #GPPdifflist=[-999]*(j-i)
    Cflist=[58]+[-999]*(j-i)
    
    for x in range(i,j):
        GPPlist[x-i]=ACM(Cflist[x-i], float(d.phi_d[x]), float(d.T_range[x]), float(d.R_tot[x]), float(d.T_max[x]), float(d.D[x]), float(d.I[x]))
        Cflist[x-i+1]=(1-d.p_5)*Cflist[x-i]+d.p_3*(1-d.p_2)*GPPlist[x-i]
    
    return GPPlist
