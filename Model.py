
import ad
import numpy as np
import Data as D
d=D.dalecData()

#ACM & DALEC model equations:
def GPP(Cf, d, x): #Gross Primary Production (GPP) function
    L=Cf/111.
    q=d.a_3-d.a_4
    gc=((abs(float(d.phi_d[x])))**(d.a_10))/(0.5*float(d.T_range[x])+d.a_6*float(d.R_tot[x]))
    p=((d.a_1*d.N*L)/gc)*np.exp(d.a_8*float(d.T_max[x]))
    Ci=0.5*(d.C_a+q-p+np.sqrt((d.C_a+q-p)**2-4*(d.C_a*q-p*d.a_3)))
    E0=(d.a_7*L**2)/(L**2+d.a_9)
    delta=-0.408*np.cos(((360*(float(d.D[x])+10)*np.pi)/(365*180)))
    s=24*np.arccos((-np.tan(d.bigdelta)*np.tan(delta)))/np.pi
    GPP=(E0*float(d.I[x])*gc*(d.C_a-Ci)*(d.a_2*s+d.a_5))/(E0*float(d.I[x])+gc*(d.C_a-Ci))    
    return GPP


def GPPdiff(Cf, d, x): #derivative of GPP
    Cf=ad.adnumber(Cf)
    L=Cf/111.
    q=d.a_3-d.a_4
    gc=((abs(float(d.phi_d[x])))**(d.a_10))/(0.5*float(d.T_range[x])+d.a_6*float(d.R_tot[x]))
    p=((d.a_1*d.N*L)/gc)*np.exp(d.a_8*float(d.T_max[x]))
    Ci=0.5*(d.C_a+q-p+np.sqrt((d.C_a+q-p)**2-4*(d.C_a*q-p*d.a_3)))
    E0=(d.a_7*L**2)/(L**2+d.a_9)
    delta=-0.408*np.cos(((360*(float(d.D[x])+10)*np.pi)/(365*180)))
    s=24*np.arccos((-np.tan(d.bigdelta)*np.tan(delta)))/np.pi
    GPP=(E0*float(d.I[x])*gc*(d.C_a-Ci)*(d.a_2*s+d.a_5))/(E0*float(d.I[x])+gc*(d.C_a-Ci))    
    return GPP.d(Cf)


def DALEC(Cf,Cr,Cw,Cl,Cs,x,d): #DALEC evergreen model equations
    GPPval=GPP(Cf, d, x)
    Cf2=(1-d.p_5)*Cf+d.p_3*(1-d.p_2)*GPPval
    Cr2=(1-d.p_7)*Cr+d.p_4*(1-d.p_3)*(1-d.p_2)*GPPval
    Cw2=(1-d.p_6)*Cw+(1-d.p_4)*(1-d.p_3)*(1-d.p_2)*GPPval    
    Cl2=(1-(d.p_1+d.p_8)*d.T[x])*Cl+d.p_5*Cf+d.p_7*Cr
    Cs2=(1-d.p_9*d.T[x])*Cs+d.p_6*Cw+d.p_1*d.T[x]*Cr    
    return [Cf2,Cr2,Cw2,Cl2,Cs2]
 
    
def LinDALEC(Cf,Cr,Cw,Cl,Cs,x,d): #Tangent Linear DALEC evergreen model
    Cf=ad.adnumber(Cf)
    Cr=ad.adnumber(Cr)
    Cw=ad.adnumber(Cw)
    Cl=ad.adnumber(Cl)
    Cs=ad.adnumber(Cs)
    Dalecoutput=DALEC(Cf,Cr,Cw,Cl,Cs,x,d)
    M=np.matrix(ad.jacobian(Dalecoutput,[Cf,Cr,Cw,Cl,Cs]))    
    return M

	
def Modlist(d,i,j): #Produces a list of carbon pool values and tangent linear matrices between two times steps (i and j) using d.Clist as initial conditions.
    Mlist=[-9999]*(j-i)
    Clist=d.Clist[:]+[-9999]*(j-i)
    for x in xrange(i,j):    
        Cf=float(Clist[x-i][0])
        Cr=float(Clist[x-i][1])
        Cw=float(Clist[x-i][2])
        Cl=float(Clist[x-i][3])
        Cs=float(Clist[x-i][4])
        Mlist[x-i]=LinDALEC(Cf,Cr,Cw,Cl,Cs,x,d)  
        Clist[(x+1)-i]=DALEC(Cf,Cr,Cw,Cl,Cs,x,d)    
    return Clist,Mlist


#Observations:
def NEE(Cf,Cr,Cw,Cl,Cs,x,d):
    NEE=-(1-d.p_2)*GPP(Cf, d, x)+d.p_8*d.T[x]*Cl+d.p_9*d.T[x]*Cs    
    return NEE

def LF(Cf,Cr,Cw,Cl,Cs,x,d):    
    LF=d.p_5*Cf
    return LF
    
def LW(Cf,Cr,Cw,Cl,Cs,x,d):
    LW=d.p_6*Cw
    return LW
    
def Cf(Cf,Cr,Cw,Cl,Cs,x,d):
    Cf=Cf
    return Cf
    
def Cr(Cf,Cr,Cw,Cl,Cs,x,d):
    Cr=Cr
    return Cr
    
def Cw(Cf,Cr,Cw,Cl,Cs,x,d):
    Cw=Cw
    return Cw
    
def Cl(Cf,Cr,Cw,Cl,Cs,x,d):
    Cl=Cl
    return Cl
    
def Cs(Cf,Cr,Cw,Cl,Cs,x,d):
    Cs=Cs
    return Cs    
