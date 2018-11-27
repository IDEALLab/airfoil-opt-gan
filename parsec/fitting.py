from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b, fmin_slsqp, differential_evolution
from synthesis import sythesize


def parsec_airfoil(airfoil):
    
    n_points = airfoil.shape[0]
    func = lambda x: np.linalg.norm(sythesize(x, n_points) - airfoil)
    bounds = [(0.001, 0.1), # rle
              (1e-4, 0.5), # x_pre
              (-0.1, 0.0), # y_pre
              (-0.5, 0.5), # d2ydx2_pre
              (-10, 10), # th_pre
              (1e-4, 0.5), # x_suc
              (0.0, 0.1), # y_suc
              (-0.5, 0.5), # d2ydx2_suc
              (-10, 10) # th_suc
              ]
    bounds = np.array(bounds)
    n_restarts = 10
    opt_x = None
    opt_f = np.inf
    x0s = np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, bounds.shape[0]))
    for x0 in x0s:
        x, f, _ = fmin_l_bfgs_b(func, x0, approx_grad=1, bounds=bounds, disp=1)
        if f < opt_f:
            opt_x = x
            opt_f = f
#    res = differential_evolution(func, bounds=bounds, disp=1)
#    opt_x = res.x
#    opt_f = res.fun
            
    print(opt_x)
    print(opt_f)
    
    return opt_x
    
if __name__ == "__main__":
    
    data_path = '../initial_airfoil/naca0012.dat'
    init_airfoil = np.loadtxt(data_path, skiprows=1)
#    x = parsec_airfoil(init_airfoil)
    x = [0.0147, 0.2996, -0.06, 0.4406, 7.335, 0.3015, 0.0599, -0.4360, -7.335] # NACA 0012
    init_airfoil_fitted = sythesize(x, n_points=192)
    
    plt.figure()
    plt.plot(init_airfoil[:,0], init_airfoil[:,1], 'bo', ms=3, alpha=0.7, label='Original Airfoil')
    plt.plot(init_airfoil_fitted[:,0], init_airfoil_fitted[:,1], 'r-', lw=1, alpha=0.7, label='PARSEC Approximation')
    plt.legend()
    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.savefig('../initial_airfoil/initial_parsec.svg')
    plt.close()
