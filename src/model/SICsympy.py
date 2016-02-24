import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def SIC():
    sigCfb,sigCrb,sigCwb,sigClb,sigCsb,sigCfo,dGPPCf,T,p1,p2,p3,p4,p5,p6,p7,p8,p9=sp.symbols("sigCfb sigCrb sigCwb sigClb sigCsb sigCfo dGPP T p1 p2 p3 p4 p5 p6 p7 p8 p9")

    B=sp.Matrix([[sigCfb**2,0,0,0,0],[0,sigCrb**2,0,0,0],[0,0,sigCwb**2,0,0],[0,0,0,sigClb**2,0],[0,0,0,0,sigCsb**2]])

    H0=sp.Matrix([[1,0,0,0,0]])
    
    M=sp.Matrix([[dGPPCf,0,0,0,0],[p4*(1-p3)*(1-p2)*dGPPCf,(1-p7),0,0,0],[(1-p4)*(1-p3)*(1-p2)*dGPPCf,0,(1-p6),0,0],[p5,p7,0,(1-(p1+p8)*T),0],[0,0,p6,p1*T,(1-p9*T)]])

    R=sp.Matrix([[sigCfo**2, 0,0,0],[0,sigCfo**2,0,0],[0,0,sigCfo**2,0],[0,0,0,sigCfo**2]])
    
    H=sp.Matrix([H0,H0*M,H0*M**2,H0*M**3])
    
    J2d=B**(-1)+H.T*(R**(-1))*H

    return sp.simplify(J2d.det()*(B).det())


def Hmat():
    """ Computes analytic rank of \hat{H} using Symbolic Python
    :return: rank of \hat{H}
    """
    # define parameters
    dGPPCf1, dGPPCf2, dGPPCf3, dGPPCf4, dGPPCf5, dGPPCf6, T1, T2, T3, T4, T5, T6, \
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, clma = sp.symbols("dGPP1 dGPP2 dGPP3 dGPP4 dGPP5 dGPP6 T1 T2 T3 T4 \
                                                    T5 T6 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 clma")
    # gpp and temperatures at different times
    gpp_lst = [dGPPCf1, dGPPCf2, dGPPCf3, dGPPCf4, dGPPCf5, dGPPCf6]
    t_lst = [T1, T2, T3, T4, T5, T6]
    # Empty lists to be filled
    H_lst = []
    M_lst = []
    H2_lst = []
    M2_lst = []
    for x in xrange(len(t_lst)):
        H_lst.append(sp.Matrix([[(1-p2)*gpp_lst[x],0,0,p8*t_lst[x],p9*t_lst[x]]]))
        # H_lst.append(sp.Matrix([[1/clma,0,0,0,0]]))
        H2_lst.append(sp.Matrix([[dGPPCf1,0,0,T1,T2]]))
        # H2_lst.append(sp.Matrix([[0,0,0,0,1]]))

        M_lst.append(sp.Matrix([[(1-p5)+p3*(1-p2)*gpp_lst[x],0,0,0,0],
                     [p4*(1-p3)*(1-p2)*gpp_lst[x],(1-p7),0,0,0],
                     [(1-p4)*(1-p3)*(1-p2)*gpp_lst[x],0,(1-p6),0,0],
                     [p5,p7,0,(1-(p1+p8)*t_lst[x]),0],
                     [0,0,p6,p1*t_lst[x],(1-p9*t_lst[x])]]))

        M2_lst.append(sp.Matrix([[p1,0,0,0,0],
                     [p2,p3,0,0,0],
                     [p4,0,p5,0,0],
                     [p6,p7,0,p8,0],
                     [0,0,p9,p10,p11]]))

    h = sp.Matrix([[(1-p2)*gpp_lst[x],0,0,p8*t_lst[x],p9*t_lst[x]]])
    M = sp.Matrix([[p1,0,0,0,0],
                   [p2,p3,0,0,0],
                   [p4,0,p5,0,0],
                   [p6,p7,0,p8,0],
                   [0,0,p9,p10,p11]])
    M2 = sp.Matrix([[(1-p5)+p3*(1-p2)*gpp_lst[x],0,0,0,0],
                     [p4*(1-p3)*(1-p2)*gpp_lst[x],(1-p7),0,0,0],
                     [(1-p4)*(1-p3)*(1-p2)*gpp_lst[x],0,(1-p6),0,0],
                     [p5,p7,0,(1-(p1+p8)*t_lst[x]),0],
                     [0,0,p6,p1*t_lst[x],(1-p9*t_lst[x])]])
    # H = sp.Matrix([H_lst[0],H_lst[1]*M_lst[0],H_lst[2]*M_lst[1]*M_lst[0],H_lst[3]*M_lst[2]*M_lst[1]*M_lst[0],
    #               H_lst[4]*M_lst[3]*M_lst[2]*M_lst[1]*M_lst[0],
    #               H_lst[5]*M_lst[4]*M_lst[3]*M_lst[2]*M_lst[1]*M_lst[0]])

    H2 = sp.Matrix([H2_lst[0],H2_lst[1]*M2_lst[0],H2_lst[2]*M2_lst[1]*M2_lst[0],H2_lst[3]*M2_lst[2]*M2_lst[1]*M2_lst[0],
                    H2_lst[4]*M2_lst[3]*M2_lst[2]*M2_lst[1]*M2_lst[0]])

    return H2, M, M2 #H2.rank(simplify=True)


def hmat_ob(obs):
    """ Computes analytic rank of \hat{H} using Symbolic Python
    :return: rank of \hat{H}
    """
    # define parameters
    dgppcf1, dgppcf2, dgppcf3, dgppcf4, dgppcf5, dgppcf6, T1, T2, T3, T4, T5, T6, p1, p2, p3, p4, p5, p6, p7, p8, \
    p9, p10, p11, p12, clma = sp.symbols("dgpp1 dgpp2 dgpp3 dgpp4 dgpp5 dgpp6 T1 T2 T3 T4 T5 T6 p1 p2 p3 p4 p5 p6 "
                                         "p7 p8 p9 p10 p11 p12 clma")
    # gpp and temperatures at different times
    gpp_lst = [dgppcf1, dgppcf2, dgppcf3, dgppcf4, dgppcf5] #, dgppcf6]
    t_lst = [T1, T2, T3, T4, T5] #, T6]
    # Empty lists to be filled
    H_lst = []
    M_lst = []
    for x in xrange(len(t_lst)):
        if obs == 'nee':
            H_lst.append(sp.Matrix([[-(1-p2)*gpp_lst[x], 0, 0, p8*t_lst[x], p9*t_lst[x]]]))
        if obs == 'lai':
            H_lst.append(sp.Matrix([[1/clma, 0, 0, 0, 0]]))
        if obs == 'cf':
            H_lst.append(sp.Matrix([[1, 0, 0, 0, 0]]))
        if obs == 'cr':
            H_lst.append(sp.Matrix([[0, 1, 0, 0, 0]]))
        if obs == 'cw':
            H_lst.append(sp.Matrix([[0, 0, 1, 0, 0]]))
        if obs == 'cl':
            H_lst.append(sp.Matrix([[0, 0, 0, 1, 0]]))
        if obs == 'cs':
            H_lst.append(sp.Matrix([[0, 0, 0, 0, 1]]))
        if obs == 'g_resp':
            H_lst.append(sp.Matrix([[(1./3.)*p2*gpp_lst[x], 0, 0, p8*t_lst[x], p9*t_lst[x]]]))
        if obs == 'lit_resp':
            H_lst.append(sp.Matrix([[0, 0, 0, p8*t_lst[x], 0]]))

        M_lst.append(sp.Matrix([[(1-p5)+p3*(1-p2)*gpp_lst[x],0,0,0,0],
                     [p4*(1-p3)*(1-p2)*gpp_lst[x],(1-p7),0,0,0],
                     [(1-p4)*(1-p3)*(1-p2)*gpp_lst[x],0,(1-p6),0,0],
                     [p5,p7,0,(1-(p1+p8)*t_lst[x]),0],
                     [0,0,p6,p1*t_lst[x],(1-p9*t_lst[x])]]))

    H = sp.Matrix([H_lst[0],H_lst[1]*M_lst[0],H_lst[2]*M_lst[1]*M_lst[0],H_lst[3]*M_lst[2]*M_lst[1]*M_lst[0],
                   H_lst[4]*M_lst[3]*M_lst[2]*M_lst[1]*M_lst[0]])
    return H.rank()

def plot_observability(obslist):
    n = len(obslist)
    width = 0.35
    ind = np.arange(n)
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    hmat_lst = [hmat_ob(obslist[x]) for x in range(len(obslist))]
    # values = [x.rank(simplify=True) for x in hmat_lst]
    values = [1, 1, 2, 2, 3, 5, 5, 5]
    ax.bar(ind, values, width, color='g')
    ax.set_ylabel(r'Rank of $\hat{\bf{H}}$')
    ax.set_title('Observability')
    ax.set_xticks(ind)
    ax.set_yticks(np.arange(24))
    keys = ['LAI', r'$C_f$', r'$C_r$', r'$C_w$', r'$C_l$', r'$C_s$', 'NEE', 'ground resp']
    ax.set_xticklabels(keys, rotation=45)
    return ax, fig, hmat_lst