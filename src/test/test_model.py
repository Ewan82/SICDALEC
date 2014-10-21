"""Tests for functions in Model.py file
"""

from model import Model as M
from model import Data as D


def test_NEE():
    dC=D.dalecData(1095)
    assert M.NEE(0,0,0,0,0,0,dC) == 0