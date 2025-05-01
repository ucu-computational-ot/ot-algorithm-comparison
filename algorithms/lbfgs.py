import numpy as np
from ot.smooth import smooth_ot_dual

def dual_lbfgs(a, b, C, epsillon=1e-3):
    P = smooth_ot_dual(a, b, C, reg=epsillon, reg_type="negentropy")
    return P, np.sum(P * C)


def dual_lbfs_potentials(a, b, C, epsilon=1e-3):
    _, log = smooth_ot_dual(a, b, C, reg=epsilon, reg_type="negentropy",
                            log=True)
    return log['alpha'], log['beta']
    
