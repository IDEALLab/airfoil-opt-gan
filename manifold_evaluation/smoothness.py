"""
Measures relative smoothness of generated shapes using relative variance of difference (RVOD)

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from utils import mean_err


def variation(X):
    var = 0
    for x in X:
        diff = np.diff(x, axis=0)
        cov = np.cov(diff.T)
        var += np.trace(cov)/cov.shape[0]
    return var/X.shape[0]
    
def ci_rsmth(n, gen_func, X_test):
    rsmth = np.zeros(n)
    for i in range(n):
        X_gen = gen_func(2000)
        rsmth[i] = variation(np.squeeze(X_test))/variation(X_gen)
    mean, err = mean_err(rsmth)
    return mean, err