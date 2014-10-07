import sympy as sp

def SIC():
    sigCfb,sigCrb,sigCwb,sigClb,sigCsb,sigCfo,dGPPCf,T,p1,p2,p3,p4,p5,p6,p7,p8,p9=sp.symbols("sigCfb sigCrb sigCwb sigClb sigCsb sigCfo dGPP T p1 p2 p3 p4 p5 p6 p7 p8 p9")

    B=sp.Matrix([[sigCfb**2,0,0,0,0],[0,sigCrb**2,0,0,0],[0,0,sigCwb**2,0,0],[0,0,0,sigClb**2,0],[0,0,0,0,sigCsb**2]])

    H0=sp.Matrix([[1,0,0,0,0]])
    
    M=sp.Matrix([[dGPPCf,0,0,0,0],[p4*(1-p3)*(1-p2)*dGPPCf,(1-p7),0,0,0],[(1-p4)*(1-p3)*(1-p2)*dGPPCf,0,(1-p6),0,0],[p5,p7,0,(1-(p1+p8)*T),0],[0,0,p6,p1*T,(1-p9*T)]])

    R=sp.Matrix([[sigCfo**2, 0,0,0],[0,sigCfo**2,0,0],[0,0,sigCfo**2,0],[0,0,0,sigCfo**2]])
    
    H=sp.Matrix([H0,H0*M,H0*M**2,H0*M**3])
    
    J2d=B**(-1)+H.T*(R**(-1))*H

    return sp.simplify(J2d.det()*(B).det())
