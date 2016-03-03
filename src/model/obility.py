import numpy as np
import Model as Mod


def Mfac(Mlist,a): #Matrix factoral to find product of M matrices
    if a == 0:
        return Mlist[0]
    else:
        Mat = Mlist[0]
        for x in xrange(0,a):
            Mat = np.dot(Mlist[x+1], Mat)
        return Mat


def Rhat(d, i, j, obs):
    oberr_dict = {'nee': d.sigO_nee, 'lai': d.sigO_lai, 'cf': d.sigO_cf, 'cr': d.sigO_cr,
                  'cw': d.sigO_cw, 'cl': d.sigO_cl, 'cs': d.sigO_cs, 'g_resp': d.sigO_g_resp,}
    Rhat = np.eye(j-i)*oberr_dict[obs]
    return np.matrix(Rhat)


def Hmat(d, i, j, obs):
    """
    Computes the observability matrix for DALEC1
    :param d: dataClass for DALEC1 Data.py
    :param i: start run
    :param j: end run
    :param obs: sting for observed value
    :return:
    """
    Hlist = [-9999]*(j-i)
    Clist = Mod.Clist(d, i, j)
    Mlist = Mod.Mlist(d,i,j)

    for x in range(i,j):
        GPPdiff = Mod.GPPdiff(Clist[x][0],d,x)
        if obs == 'nee':
            Hlist[x-i] = np.matrix([[-(1.-d.p_2)*GPPdiff, 0, 0, d.p_8*d.T[x], d.p_9*d.T[x]]])
        if obs == 'lai':
            Hlist[x-i] = np.matrix([[1./111., 0, 0, 0, 0]])
        if obs == 'cf':
            Hlist[x-i] = np.matrix([[1., 0, 0, 0, 0]])
        if obs == 'cr':
            Hlist[x-i] = np.matrix([[0, 1., 0, 0, 0]])
        if obs == 'cw':
            Hlist[x-i] = np.matrix([[0, 0, 1., 0, 0]])
        if obs == 'cl':
            Hlist[x-i] = np.matrix([[0, 0, 0, 1., 0]])
        if obs == 'cs':
            Hlist[x-i] = np.matrix([[0, 0, 0, 0, 1.]])
        if obs == 'g_resp':
            Hlist[x-i] = np.matrix([[(1./3.)*d.p_2*GPPdiff, 0, 0, d.p_8*d.T[x], d.p_9*d.T[x]]])

    stacklist = [Hlist[0]]+[-9999]*(j-i-1) #Creates H hat matrix
    for x in range(1,j-i):
        stacklist[x]=Hlist[x]*Mfac(Mlist,x-1)

    Hmat = np.vstack(stacklist)

    rhat = Rhat(d, i, j, obs)
    
    return Hmat


def Obility(d, i, j, obs):
    H = Hmat(d, i, j, obs)
    return np.linalg.matrix_rank(H)
