import numpy as np
from nurbs.bspline import eval_error
from nurbs.wc_least_squares import wc_least_squares_curve


def fit_single_curve(Q, n_control_points, p, upper=True):
    # Order: leading --> trailing edge
    n = n_control_points-1
    n_points = Q.shape[0]
    Wq = np.ones(n_points)
    Wq[0] = Wq[-1] = -1 # fix leading and trailing edge
    D = np.array([[0,-1]]) * (-1)**upper
    I = np.array([0])
    Wd = np.array([-1])
    P, U = wc_least_squares_curve(Q, Wq, D, I, Wd, n, p)
    W = np.ones(n+1)
    err = eval_error(Q, P, W, U, p)
    print 'error:', err
    return P, U

def nurbs_airfoil(n_control_points, p, coordinates):
    if type(coordinates) == str:
        airfoil = np.loadtxt(coordinates, skiprows=1)
    else:
        airfoil = coordinates
    n_points = airfoil.shape[0]
    upper = np.flip(airfoil[:n_points/2], axis=0)
    lower = airfoil[n_points/2:]
    P_upper, U_upper = fit_single_curve(upper, n_control_points, p, upper=True)
    P_lower, U_lower = fit_single_curve(lower, n_control_points, p, upper=False)
    x0 = np.concatenate(([P_upper[1,1]], P_upper[2:-1].flatten(), [P_lower[1,1]], P_lower[2:-1].flatten()))    
    return x0, U_upper, U_lower