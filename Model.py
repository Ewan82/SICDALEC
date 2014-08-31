#ACM model for GPP
import ad
import numpy as np
import Data as D
d=D.dalecData()


def GPP(Cf, phi_d, T_range, R_tot, T_max, D, I):
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

def GPPdiff(Cf, phi_d, T_range, R_tot, T_max, D, I):
    Cf=ad.adnumber(Cf)
    L=Cf/111.
    q=d.a_3-d.a_4
    gc=((abs(phi_d))**(d.a_10))/(0.5*T_range+d.a_6*R_tot)
    p=((d.a_1*d.N*L)/gc)*np.exp(d.a_8*T_max)
    Ci=0.5*(d.C_a+q-p+np.sqrt((d.C_a+q-p)**2-4*(d.C_a*q-p*d.a_3)))
    E0=(d.a_7*L**2)/(L**2+d.a_9)
    delta=-0.408*np.cos(((360*(D+10)*np.pi)/(365*180)))
    s=24*np.arccos((-np.tan(d.bigdelta)*np.tan(delta)))/np.pi
    GPP=(E0*I*gc*(d.C_a-Ci)*(d.a_2*s+d.a_5))/(E0*I+gc*(d.C_a-Ci))
    
    return GPP.d(Cf)