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
    return GPPlist ,GPPdifflist


"""GPPdiff
    -5.67694353518161e-8*Cf**3*I*(0.3408*acos(1.281073506194*tan(0.408*cos(pi*(2*D/3
65 + 4/73))))/pi + 0.155)*(0.0262094594594595*Cf*(2.653*R_tot + 0.5*T_range)*exp
(0.06*T_max)*Abs(phi_d)**-0.0006 - 0.5*(45.6883297297297*Cf*(2.653*R_tot + 0.5*T
_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + (-0.0524189189189189*Cf*(2.653*R_t
ot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + 571.92)**2 - 308026.4)**
(1/2) + 69.04)*Abs(phi_d)**0.0006/((8.11622433244055e-5*Cf**2 + 1.062)**2*(2.653
*R_tot + 0.5*T_range)*(0.000349728106484863*Cf**2*I/(8.11622433244055e-5*Cf**2 +
 1.062) + (0.0262094594594595*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs
(phi_d)**-0.0006 - 0.5*(45.6883297297297*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06
*T_max)*Abs(phi_d)**-0.0006 + (-0.0524189189189189*Cf*(2.653*R_tot + 0.5*T_range
)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + 571.92)**2 - 308026.4)**(1/2) + 69.04)*A
bs(phi_d)**0.0006/(2.653*R_tot + 0.5*T_range))) + 0.000349728106484863*Cf**2*I*(
0.3408*acos(1.281073506194*tan(0.408*cos(pi*(2*D/365 + 4/73))))/pi + 0.155)*(0.0
262094594594595*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 
- 0.5*(-0.0524189189189189*(2.653*R_tot + 0.5*T_range)*(-0.0524189189189189*Cf*(
2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + 571.92)*exp(0.0
6*T_max)*Abs(phi_d)**-0.0006 + 22.8441648648649*(2.653*R_tot + 0.5*T_range)*exp(
0.06*T_max)*Abs(phi_d)**-0.0006)/(45.6883297297297*Cf*(2.653*R_tot + 0.5*T_range
)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + (-0.0524189189189189*Cf*(2.653*R_tot + 0
.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + 571.92)**2 - 308026.4)**(1/2))
*Abs(phi_d)**0.0006/((8.11622433244055e-5*Cf**2 + 1.062)*(2.653*R_tot + 0.5*T_ra
nge)*(0.000349728106484863*Cf**2*I/(8.11622433244055e-5*Cf**2 + 1.062) + (0.0262
094594594595*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 
- 0.5*(45.6883297297297*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d
)**-0.0006 + (-0.0524189189189189*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)
*Abs(phi_d)**-0.0006 + 571.92)**2 - 308026.4)**(1/2) + 69.04)*Abs(phi_d)**0.0006
/(2.653*R_tot + 0.5*T_range))) + 0.000349728106484863*Cf**2*I*(0.3408*acos(1.281
073506194*tan(0.408*cos(pi*(2*D/365 + 4/73))))/pi + 0.155)*(5.67694353518161e-8*
Cf**3*I/(8.11622433244055e-5*Cf**2 + 1.062)**2 - 0.000699456212969726*Cf*I/(8.11
622433244055e-5*Cf**2 + 1.062) - (0.0262094594594595*(2.653*R_tot + 0.5*T_range)
*exp(0.06*T_max)*Abs(phi_d)**-0.0006 - 0.5*(-0.0524189189189189*(2.653*R_tot + 0
.5*T_range)*(-0.0524189189189189*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*
Abs(phi_d)**-0.0006 + 571.92)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + 22.844164864
8649*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006)/(45.688329
7297297*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + (-0
.0524189189189189*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.
0006 + 571.92)**2 - 308026.4)**(1/2))*Abs(phi_d)**0.0006/(2.653*R_tot + 0.5*T_ra
nge))*(0.0262094594594595*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi
_d)**-0.0006 - 0.5*(45.6883297297297*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_m
ax)*Abs(phi_d)**-0.0006 + (-0.0524189189189189*Cf*(2.653*R_tot + 0.5*T_range)*ex
p(0.06*T_max)*Abs(phi_d)**-0.0006 + 571.92)**2 - 308026.4)**(1/2) + 69.04)*Abs(p
hi_d)**0.0006/((8.11622433244055e-5*Cf**2 + 1.062)*(2.653*R_tot + 0.5*T_range)*(
0.000349728106484863*Cf**2*I/(8.11622433244055e-5*Cf**2 + 1.062) + (0.0262094594
594595*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 - 0.5*
(45.6883297297297*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.
0006 + (-0.0524189189189189*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(p
hi_d)**-0.0006 + 571.92)**2 - 308026.4)**(1/2) + 69.04)*Abs(phi_d)**0.0006/(2.65
3*R_tot + 0.5*T_range))**2) + 0.000699456212969726*Cf*I*(0.3408*acos(1.281073506
194*tan(0.408*cos(pi*(2*D/365 + 4/73))))/pi + 0.155)*(0.0262094594594595*Cf*(2.6
53*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 - 0.5*(45.6883297297
297*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + (-0.052
4189189189189*Cf*(2.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006
 + 571.92)**2 - 308026.4)**(1/2) + 69.04)*Abs(phi_d)**0.0006/((8.11622433244055e
-5*Cf**2 + 1.062)*(2.653*R_tot + 0.5*T_range)*(0.000349728106484863*Cf**2*I/(8.1
1622433244055e-5*Cf**2 + 1.062) + (0.0262094594594595*Cf*(2.653*R_tot + 0.5*T_ra
nge)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 - 0.5*(45.6883297297297*Cf*(2.653*R_tot
 + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + (-0.0524189189189189*Cf*(2
.653*R_tot + 0.5*T_range)*exp(0.06*T_max)*Abs(phi_d)**-0.0006 + 571.92)**2 - 308
026.4)**(1/2) + 69.04)*Abs(phi_d)**0.0006/(2.653*R_tot + 0.5*T_range)))"""
