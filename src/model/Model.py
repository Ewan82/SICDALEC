"""
Put a docstring here.
"""

import numpy as np
import ad

#ACM & DALEC model equations:
def GPP(Cf, d, x):
    """Gross Primary Production (GPP) function
    ------------------------------------------
    Takes a foliar carbon (Cf) value, a DataClass (d) and a time step (x) and 
    returns the estimated value for GPP of the forest.
    """
    L = Cf / 111.
    q = d.a_3 - d.a_4
    gc = (abs(d.phi_d[x]))**(d.a_10) / (0.5*d.T_range[x] + d.a_6*d.R_tot[x])
    p = ((d.a_1*d.N*L) / gc)*np.exp(d.a_8*d.T_max[x])
    Ci = 0.5*(d.C_a + q - p + np.sqrt((d.C_a + q - p)**2 \
         -4*(d.C_a*q - p*d.a_3)))
    E0 = (d.a_7*L**2) / (L**2 + d.a_9)
    delta = -0.408*np.cos((360*(d.D[x] + 10)*np.pi) / (365*180))
    s = 24*np.arccos(( - np.tan(d.bigdelta)*np.tan(delta))) / np.pi
    GPP = (E0*d.I[x]*gc*(d.C_a - Ci)*(d.a_2*s + d.a_5)) / \
          (E0*d.I[x] + gc*(d.C_a - Ci))    
    return GPP


def GPPdiff(Cf, d, x): 
    """Gross Primary Production derivative (GPP) function
    -----------------------------------------------------
    Takes a foliar carbon (Cf) value, a DataClass (d) and a time step (x) and 
    returns the estimated value for the derivative of the GPP of the forest.
    Uses the module ad for automatic differentiation.
    """
    Cf = ad.adnumber(Cf)
    L = Cf / 111.
    q = d.a_3 - d.a_4
    gc = (abs(d.phi_d[x]))**(d.a_10) / (0.5*d.T_range[x] + d.a_6*d.R_tot[x])
    p = ((d.a_1*d.N*L) / gc)*np.exp(d.a_8*d.T_max[x])
    Ci = 0.5*(d.C_a + q - p + np.sqrt((d.C_a + q - p)**2 \
         -4*(d.C_a*q - p*d.a_3)))
    E0 = (d.a_7*L**2) / (L**2 + d.a_9)
    delta = -0.408*np.cos(((360*(d.D[x] + 10)*np.pi) / (365*180)))
    s = 24*np.arccos(( - np.tan(d.bigdelta)*np.tan(delta))) / np.pi
    GPP = (E0*d.I[x]*gc*(d.C_a - Ci)*(d.a_2*s + d.a_5)) / \
          (E0*d.I[x] + gc*(d.C_a - Ci))    
    return GPP.d(Cf)


def DALEC(Cf, Cr, Cw, Cl, Cs, x, d): #DALEC evergreen model equations
    GPPval = GPP(Cf, d, x)
    Cf2 = (1-d.p_5)*Cf + d.p_3*(1-d.p_2)*GPPval
    Cr2 = (1-d.p_7)*Cr + d.p_4*(1-d.p_3)*(1-d.p_2)*GPPval
    Cw2 = (1-d.p_6)*Cw + (1-d.p_4)*(1-d.p_3)*(1-d.p_2)*GPPval    
    Cl2 = (1-(d.p_1 + d.p_8)*d.T[x])*Cl + d.p_5*Cf + d.p_7*Cr
    Cs2 = (1 - d.p_9*d.T[x])*Cs + d.p_6*Cw + d.p_1*d.T[x]*Cr    
    return np.array([Cf2, Cr2, Cw2, Cl2, Cs2])
 
    
def LinDALEC(Cf, Cr, Cw, Cl, Cs, x, d): #Tangent Linear DALEC evergreen model
    Cf = ad.adnumber(Cf)
    Cr = ad.adnumber(Cr)
    Cw = ad.adnumber(Cw)
    Cl = ad.adnumber(Cl)
    Cs = ad.adnumber(Cs)
    Dalecoutput = DALEC(Cf, Cr, Cw, Cl, Cs, x, d)
    M = np.matrix(ad.jacobian(Dalecoutput, [Cf, Cr, Cw, Cl, Cs]))    
    return Dalecoutput, M

	
def Mlist(d,i,j): #Produces a list of carbon pool values and tangent linear matrices between two times steps (i and j) using d.Clist as initial conditions.
    Clist = np.concatenate((d.Clist, np.ones((j - i,5))*-9999.))
    Matlist = np.ones((j - i,5,5))*-9999.
    for x in xrange(i, j):    
        Cf = float(Clist[x - i][0])
        Cr = float(Clist[x - i][1])
        Cw = float(Clist[x - i][2])
        Cl = float(Clist[x - i][3])
        Cs = float(Clist[x - i][4])
        Clist[(x + 1) - i], Matlist[x - i] = LinDALEC(Cf, Cr, Cw, Cl, Cs, x, d) 
    return Matlist

    
def Clist(d,i,j): #Produces a list of carbon pool values and tangent linear matrices between two times steps (i and j) using d.Clist as initial conditions.
    Clist = np.concatenate((d.Clist, np.ones((j - i,5))*-9999.))
    for x in xrange(i, j):    
        Cf = float(Clist[x - i][0])
        Cr = float(Clist[x - i][1])
        Cw = float(Clist[x - i][2])
        Cl = float(Clist[x - i][3])
        Cs = float(Clist[x - i][4]) 
        Clist[(x + 1) - i] = DALEC(Cf, Cr, Cw, Cl, Cs, x, d)    
    return Clist


def Clist_lin(d,i,j):
    Clist = np.concatenate((d.Clist, np.ones((j - i,5))*-9999.))
    Matlist = Mlist(d,i,j)
    for x in xrange(i, j):
        Clist[x-i+1] = (Matlist[x-i]*np.matrix(Clist[x-i]).T).T
    return Clist


def Mfac(Mlist,a): #Matrix factoral to find product of M matrices
    if a==0:
        return np.matrix(Mlist[0])
    else:
        Mat=np.matrix(Mlist[0])
        for x in xrange(0,a):
            Mat=np.matrix(Mlist[x+1])*Mat
        return Mat
        

def Clist_lin2(d,i,j):
    Clist = np.concatenate((d.Clist, np.ones((j - i,5))*-9999.))
    Matlist = Mlist(d,i,j)
    for x in xrange(i, j):
        Clist[x-i+1] = (Mfac(Matlist,x-i)*np.matrix(Clist[0]).T).T
    return Clist


#Observations:
def NEE(Cf,Cr,Cw,Cl,Cs,x,d):
    NEE = -(1-d.p_2)*GPP(Cf, d, x) + d.p_8*d.T[x]*Cl + d.p_9*d.T[x]*Cs    
    return NEE

def LF(Cf,Cr,Cw,Cl,Cs,x,d):    
    LF = d.p_5*Cf
    return LF
    
def LW(Cf,Cr,Cw,Cl,Cs,x,d):
    LW = d.p_6*Cw
    return LW
    
def Cf(Cf,Cr,Cw,Cl,Cs,x,d):
    Cf = Cf
    return Cf
    
def Cr(Cf,Cr,Cw,Cl,Cs,x,d):
    Cr = Cr
    return Cr
    
def Cw(Cf,Cr,Cw,Cl,Cs,x,d):
    Cw = Cw
    return Cw
    
def Cl(Cf,Cr,Cw,Cl,Cs,x,d):
    Cl = Cl
    return Cl
    
def Cs(Cf,Cr,Cw,Cl,Cs,x,d):
    Cs = Cs
    return Cs    