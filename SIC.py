#This will have code in
import sympy as sp
import numpy as np
import Data as D
d=D.dalecData()

def SIC(i,j):

    #ACM model
    Cf, phi_d, T_range, R_tot, T_max, D, I=sp.symbols("Cf phi_d T_range R_tot T_max D I")
    L=Cf/111.
    q=d.a_3-d.a_4
    gc=((abs(phi_d))**(d.a_10))/(0.5*T_range+d.a_6*R_tot)
    p=((d.a_1*d.N*L)/gc)*sp.exp(d.a_8*T_max)
    Ci=0.5*(d.C_a+q-p+sp.sqrt((d.C_a+q-p)**2-4*(d.C_a*q-p*d.a_3)))
    E0=(d.a_7*L**2)/(L**2+d.a_9)
    delta=-0.408*sp.cos(((360*(D+10)*sp.pi)/(365*180)))
    s=24*sp.acos((-sp.tan(d.bigdelta)*sp.tan(delta)))/sp.pi
    GPP=(E0*I*gc*(d.C_a-Ci)*(d.a_2*s+d.a_5))/(E0*I+gc*(d.C_a-Ci))
    GPPdiff=sp.diff(GPP,Cf)
    
    GPPlist=[-999]*(j-i)
    GPPdifflist=[-999]*(j-i)
    Cflist=[58]+[-999]*(j-i)
    
    for x in range(i,j):
        GPPlist[x-i]=GPP.subs([[Cf,Cflist[x-i]],[phi_d,float(d.phi_d[x])],[T_range,float(d.T_range[x])],[R_tot,float(d.R_tot[x])],[T_max,float(d.T_max[x])],[D,float(d.D[x])],[I,float(d.I[x])]]).evalf()
        GPPdifflist[x-i]=GPPdiff.subs([[Cf,Cflist[x-i]],[phi_d,float(d.phi_d[x])],[T_range,float(d.T_range[x])],[R_tot,float(d.R_tot[x])],[T_max,float(d.T_max[x])],[D,float(d.D[x])],[I,float(d.I[x])]]).evalf()
        Cflist[x-i+1]=(1-d.p_5)*Cflist[x-i]+d.p_3*(1-d.p_2)*GPPlist[x-i]
    return GPPlist,GPPdifflist
