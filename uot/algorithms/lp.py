import ot
import numpy as np

def pot_lp(a, b, C):
    T = ot.emd(a, b, C)
    return T, np.sum(T * C)