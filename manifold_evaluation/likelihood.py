"""
Estimates likelihood of generated data using kernel density estimation 

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from utils import optimize_kde, mean_err

def mean_log_likelihood(X_gen, X_test):
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    kde = optimize_kde(X_gen)
    return kde.score(X_test)/X_test.shape[0]
#    scores = kde.score_samples(X_test)
#    return np.sum(np.exp(scores))
    
def ci_mll(n, gen_func, X_test):
    mlls = np.zeros(n)
    for i in range(n):
        X_gen = gen_func(2000)
        mlls[i] = mean_log_likelihood(X_gen, np.squeeze(X_test))
    mean, err = mean_err(mlls)
    return mean, err