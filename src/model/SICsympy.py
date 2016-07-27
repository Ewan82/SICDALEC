import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Data as D
import Model as M


def info_content(sic_dfs):
    sigCfb, sigCrb, sigCwb, sigClb, sigCsb, sigCo, sigNEEo, cor, dGPP, T, p1, p2, p3, p4, p5, p6, p7, p8, p9 = \
        sp.symbols("sigCfb sigCrb sigCwb sigClb sigCsb sigCo sigNEEo cor dGPP T p1 p2 p3 p4 p5 p6 p7 p8 p9")

    B = sp.Matrix([[sigCfb**2, 0, 0, 0, 0], [0, sigCrb**2, 0, 0, 0], [0, 0, sigCwb**2, 0, 0],
                   [0, 0, 0, sigClb**2, 0], [0, 0, 0, 0, sigCsb**2]])
    # B = sp.Matrix([[sigCfb, cor, cor, cor, cor], [cor, sigCrb, cor, cor, cor], [cor, cor, sigCwb, cor, cor],
    #               [cor, cor, cor, sigClb, cor], [cor, cor, cor, cor, sigCsb]])

    # H0 = sp.Matrix([[1, 0, 0, 1, 1]])
    H0 = sp.Matrix([[-(1-p2)*dGPP, 0, 0, p8*T, p9*T]])
    # H0 = sp.Matrix([[-(1-p2)*dGPP, 0, 0, p8*T, p9*T],
    #               [1., 0, 0, 0, 0]])

    M = sp.Matrix([[dGPP, 0, 0, 0, 0], [p4*(1-p3)*(1-p2)*dGPP, (1-p7), 0, 0, 0],
                   [(1-p4)*(1-p3)*(1-p2)*dGPP, 0, (1-p6), 0, 0],
                   [p5, p7, 0, (1-(p1+p8)*T), 0], [0, 0, p6, p1*T, (1-p9*T)]])

    # R = sp.Matrix([[sigCo**2, 0, 0, 0], [0, sigCo**2, 0, 0], [0, 0, sigCo**2, 0], [0, 0, 0, sigCo**2]])
    # R = sigCo
    # R = sigNEEo
    R_stdev = sp.Matrix([[sigCo, 0], [0, sigCo]])
    cormat = sp.Matrix([[1, cor], [cor, 1]])
    # cormat = sp.Matrix([[1, 0], [0, 1]])
    R = R_stdev*cormat*R_stdev.T

    H = sp.Matrix([H0, H0*M])
    # H = sp.Matrix(H0)

    J2d = B**(-1) + H.T*(R**(-1))*H
    # A = J2d**(-1)
    # K = A*H.T*(R**(-1))
    if sic_dfs == 'sic':
        return J2d.det()*B.det()  # , R, sp.simplify(K.T*H.T)
    elif sic_dfs == 'dfs':
        A = J2d**(-1)
        return 5. - sp.trace(B**(-1)*A)


def eval_info_content(sym_exp, sic_dfs='sic', corr=0.):
    sigCfb, sigCrb, sigCwb, sigClb, sigCsb, sigCo, sigNEEo, cor, dGPP, T, p1, p2, p3, p4, p5, p6, p7, p8, p9 = \
        sp.symbols("sigCfb sigCrb sigCwb sigClb sigCsb sigCo sigNEEo cor dGPP T p1 p2 p3 p4 p5 p6 p7 p8 p9")
    d = D.dalecData(365)
    #dGPP_val = M.GPPdiff(d.Cf, d, 0)
    dGPP_val = M.GPPdiff(200, d, 30)
    symbols = [sigCfb, sigCrb, sigCwb, sigClb, sigCsb, sigCo, sigNEEo, cor, dGPP, T, p1, p2, p3, p4, p5, p6, p7,
               p8, p9]
    diagB = d.B2.diagonal()
    #eval_vars = zip(symbols, [d.sigB_cf, d.sigB_cr, d.sigB_cw, d.sigB_cl, d.sigB_cs, d.sigO_cf, d.sigO_nee, corr,
    #                          dGPP_val, d.T[0], d.p_1, d.p_2, d.p_3, d.p_4, d.p_5, d.p_6, d.p_7, d.p_8, d.p_9])
    eval_vars = zip(symbols, [diagB[0,0], diagB[0,1], diagB[0,2], diagB[0,3], diagB[0,4], d.sigO_cf, d.sigO_nee, corr,
                              dGPP_val, d.T[0], d.p_1, d.p_2, d.p_3, d.p_4, d.p_5, d.p_6, d.p_7, d.p_8, d.p_9])
    out = float(sym_exp.subs(eval_vars))
    if sic_dfs == 'sic':
        return 0.5*np.log(out)
    elif sic_dfs == 'dfs':
        return out


def corr_sic(sic_dfs):
    corrs = np.linspace(0, 0.9, 9)
    sym_exp = info_content(sic_dfs)
    sic_list = [eval_info_content(sym_exp, sic_dfs, cor) for cor in corrs]
    return sic_list


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
    gpp_lst = [dgppcf1, dgppcf2, dgppcf3, dgppcf4, dgppcf5, dgppcf6]
    t_lst = [T1, T2, T3, T4, T5, T6]
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

        #M_lst.append(sp.Matrix([[(1-p5)+p3*(1-p2)*gpp_lst[x],0,0,0,0],
        #             [p4*(1-p3)*(1-p2)*gpp_lst[x],(1-p7),0,0,0],
        #             [(1-p4)*(1-p3)*(1-p2)*gpp_lst[x],0,(1-p6),0,0],
        #             [p5,p7,0,(1-(p1+p8)*t_lst[x]),0],
        #             [0,0,p6,p1*t_lst[x],(1-p9*t_lst[x])]]))
        M_lst.append(sp.Matrix([[gpp_lst[x],0,0,0,0],
                     [p4*(1-p3)*(1-p2)*gpp_lst[x],(1-p7),0,0,0],
                     [(1-p4)*(1-p3)*(1-p2)*gpp_lst[x],0,(1-p6),0,0],
                     [p5,p7,0,(1-(p1+p8)*t_lst[x]),0],
                     [0,0,p6,p1*t_lst[x],(1-p9*t_lst[x])]]))

    # H = sp.Matrix([H_lst[0], H_lst[1]*M_lst[0], H_lst[2]*M_lst[1]*M_lst[0], H_lst[3]*M_lst[2]*M_lst[1]*M_lst[0],
    #               H_lst[4]*M_lst[3]*M_lst[2]*M_lst[1]*M_lst[0],
    #               H_lst[5]*M_lst[4]*M_lst[3]*M_lst[2]*M_lst[1]*M_lst[0]])
    H = sp.Matrix([H_lst[0], H_lst[1]*M_lst[0]])
    return H_lst[0]


def sic_ob(obs):
    sigcfb, sigcrb, sigcwb, sigclb, sigcsb, sigo_11, sigo_22, sigo_12, sigo_21, a, T, p8, p9 = sp.symbols("sigcfb "
                                                                                            "sigcrb sigcwb "
                                                                                            "sigclb sigcsb sigo_11 "
                                                                                            "sigo_22 sigo_12 sigo_21 "
                                                                                            "a T p8 p9")


    B = sp.Matrix([[sigcfb**2, 0, 0, 0, 0], [0, sigcrb**2, 0, 0, 0], [0, 0, sigcwb**2, 0, 0],
                   [0, 0, 0, sigclb**2, 0], [0, 0, 0, 0, sigcsb**2]])

    Jdd = sp.Matrix([[sigcfb**-2+(sigo_11**-2)*a**2, 0, 0, sigo_11**-2 * a*p8*T, sigo_11**-2 *a *p9*T],
                     [0, sigcrb**-2, 0, 0, 0],
                     [0, 0, sigcwb**-2, 0, 0],
                     [sigo_11**-2 *a*p8*T, 0, 0, sigclb**-2 +sigo_11**-2*p8**2*T**2, sigo_11**-2*p8*p9*T**2],
                     [sigo_11**-2*a*p9*T, 0, 0, sigo_11**-2*p8*p9*T**2, sigcsb**-2+sigo_11**-2*p9**2*T**2]])

    H = hmat_ob(obs)

    R = sigo_11**2 * sp.eye(H.shape[0])
    # R = sp.Matrix([[sigo_11, sigo_12], [sigo_21, sigo_22]])

    J2d = B**(-1) + H.T * (R**(-1)) * H

    sic = J2d.det() * (B).det()

    dfs = 5 - (B.inv()*J2d.inv()).trace()

    return sp.simplify(sic), sp.simplify(dfs)


def plot_observability(obslist):
    sns.set(style="whitegrid")
    sns.set_context('paper', font_scale=1.4)
    n = len(obslist)
    width = 0.35
    ind = np.arange(n)
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    # hmat_lst = [hmat_ob(obslist[x]) for x in range(len(obslist))]
    # values = [x.rank(simplify=True) for x in hmat_lst]
    values = [1, 1, 2, 2, 3, 5, 5, 5]
    ax.bar(ind, values, width, color='g')
    ax.set_ylabel(r'Rank of $\hat{\bf{H}}$')
    # ax.set_title('Observability')
    ax.set_xticks(ind)
    ax.set_yticks(np.arange(6))
    keys = ['LAI', r'$C_{fol}$', r'$C_{roo}$', r'$C_{woo}$', r'$C_{lit}$', r'$C_{som}$', 'NEE', 'ground resp']
    ax.set_xticklabels(keys, rotation=45)
    return ax, fig, # hmat_lst


def fiveinv():
    a11, a14, a15, a22, a33, a41, a44, a45, a51, a54, a55 = sp.symbols("a11 a14 a15 a22 a33 a41 a44 a45 "
                                                                       "a51 a54 a55")
    d2J = sp.Matrix([[a11, 0, 0, a14, a14],
                     [0, a22, 0, 0, 0],
                     [0, 0, a33, 0, 0],
                     [a41, 0, 0, a44, a45],
                     [a51, 0, 0, a54, a55]])

    return d2J


def corr_eff():
    xb, x0, y0, y1, sigyo0, sigyo1, cor, sigxb, a = sp.symbols("xb x0 y0 y1 sigyo0 sigyo1 cor sigxb, a")

    B = sp.Matrix([[sigxb**2]])

    H0 = sp.Matrix([[1]])

    M = sp.Matrix([[a]])

    R_stdev = sp.Matrix([[sigyo0, 0], [0, sigyo0]])
    cormat = sp.Matrix([[1, cor], [cor, 1]])
    # cormat = sp.Matrix([[1, 0], [0, 1]])
    R = R_stdev*cormat*R_stdev.T

    H = sp.Matrix([H0, H0*M])

    yhat = sp.Matrix([[y0, y1]]).T

    Hx0 = H*x0

    J = sp.simplify((yhat-Hx0).T*R**(-1)*(yhat-Hx0))
    J2d = B**(-1) + H.T*(R**(-1))*H
    A = J2d**(-1)
    K = A*H.T*(R**(-1))
    S = sp.simplify(K*H)
    return sp.simplify(J2d.det())*(B.det())  # sp.simplify(K)
