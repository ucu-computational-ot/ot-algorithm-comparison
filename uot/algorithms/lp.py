import ot
import numpy as np

def pot_lp(a, b, C):
    a = np.asarray(a)
    b = np.asarray(b)
    C = np.asarray(C)
    T, log = ot.emd(a, b, C, log=True, numItermax=10000000)
    if log['warning'] is not None:
        print(log['warning'])
    converged = log['warning'] is None 
    return T, np.sum(T * C), converged