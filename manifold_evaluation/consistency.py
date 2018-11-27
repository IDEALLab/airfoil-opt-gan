"""
Estimates likelihood of generated data using kernel density estimation 

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from utils import mean_err

def sample_line(d, m, bounds):
    # Sample m points along a line parallel to a d-dimensional space's basis
    basis = np.random.choice(d)
    c = np.zeros((m, d))
    c[:,:] = np.random.rand(d)
    c[:,basis] = np.linspace(0.0, 1.0, m)
    c = bounds[0] + (bounds[1]-bounds[0])*c
    return c

def consistency(gen_func, d, bounds):
    
    n_eval = 100
    n_points = 50
    mean_cor = 0
    
    for i in range(n_eval):
        
        c = sample_line(d, n_points, bounds)
        dist_c = np.linalg.norm(c - c[0], axis=1)
        
#        from matplotlib import pyplot as plt
#        plt.scatter(c[:,0], c[:,1])
        
        X = gen_func(c)
        X = X.reshape((n_points, -1))
        dist_X = np.linalg.norm(X - X[0], axis=1)
        
        mean_cor += np.corrcoef(dist_c, dist_X)[0,1]
        
    return mean_cor/n_eval

def ci_cons(n, gen_func, d=2, bounds=(0.0, 1.0)):
    conss = np.zeros(n)
    for i in range(n):
        conss[i] = consistency(gen_func, d, bounds)
    mean, err = mean_err(conss)
    return mean, err