from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def normalize(y):
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y-y_mean)/y_std
    return y

def expected_improvement(samples, gp_model, previous_optimum):
    if samples.ndim == 1:
        samples = samples.reshape(1,-1)
    mu, sigma = gp_model.predict(samples, return_std=True)
    with np.errstate(divide='ignore'):
        Z = (mu - previous_optimum)/sigma
        EI = (mu - previous_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma==0.0] = 0.0
    return EI

def neg_expected_improvement(samples, gp_model, previous_optimum):
    EI = expected_improvement(samples, gp_model, previous_optimum)
    return -EI

def sample_next_point(d, acquisition_func, gp_model, previous_optimum, bounds=None, n_restarts=25, random_search=False, 
                      previous_point=None, scale=1.0):
    assert bounds is not None or previous_point is not None
    
    opt_x = None
    opt_acquisition = 1
    
    if random_search:
        if bounds is None and previous_point is not None:
            x_random = np.random.normal(previous_point, scale, size=(random_search, d))
        else:
            x_random = np.random.uniform(bounds[:,0], bounds[:,1], size=(random_search, d))
        EI = acquisition_func(x_random, gp_model, previous_optimum)
        opt_x = x_random[np.argmin(EI)]
    else:
        for starting_point in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, d)):
            res = minimize(fun=acquisition_func,
                           x0=starting_point,
                           bounds=bounds,
                           method='L-BFGS-B',
                           args=(gp_model, previous_optimum))
            if res.fun < opt_acquisition:
                opt_acquisition = res.fun
                opt_x = res.x
            
    return opt_x