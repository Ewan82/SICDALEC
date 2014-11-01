"""Tests for functions in Model.py file
"""

import numpy as np
from model import Model as M
from model import Data as D



def test_nee():
    dC=D.dalecData(1095)
    assert M.NEE(0,0,0,0,0,0,dC) == 0
    
    
def test_gpp_derivative():
    datx = D.dalecData(1096)
    datdx = D.dalecData(1096)
    datdx.Cf = datdx.Cf*0.1
    lam = 0.00001
    mxdxgpp = M.GPP(datx.Cf+lam*datdx.Cf, datx, 0)
    mxgpp = M.GPP(datx.Cf, datx, 0)
    Mgpp = M.GPPdiff(datx.Cf, datx, 0)    
    print abs((mxdxgpp-mxgpp)/(Mgpp*lam*datdx.Cf)-1)
    assert abs(((mxdxgpp-mxgpp) / (Mgpp*lam*datdx.Cf))-1) < 1e-7
    
    
def test_lin_model():
    datx = D.dalecData(1096)
    datdx = D.dalecData(1096)
    datxdx = D.dalecData(1096)
    lam = 0.0001
    datdx.Clist = lam*0.1*datdx.Clist
    datxdx.Clist = datxdx.Clist+lam*0.1*datxdx.Clist
    mxdxgpp = M.Clist(datxdx, 0, 10)
    mxgpp = M.Clist(datx, 0, 10)
    Matlist = M.Mlist(datx, 0, 10)
    Mgpp = M.Clist_lin(datdx, 0, 10, Matlist)    
    print abs(np.linalg.norm(mxdxgpp[10]-mxgpp[10]) /  \
              np.linalg.norm(Mgpp[10]) - 1)
    assert abs((np.linalg.norm(mxdxgpp[10]-mxgpp[10]) / \
                np.linalg.norm(Mgpp[10]))-1) < 1e-7