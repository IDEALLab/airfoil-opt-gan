import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz\

from simulation import compute_coeff
from nurbs.bspline import numerical_curvature, eval_curve


def x2ctrlpts_single_curve(x):
    control_points = np.concatenate((np.zeros(3), x, np.array([1.0, 0.0])))
    control_points = control_points.reshape(-1,2)
    return control_points

def ctrlpts2x_single_curve(control_points):
    x = np.concatenate(([control_points[1,1]], control_points[2:-1].flatten()))
    return x

def new_ub_single_curve(points, ub, D=10.0):
    cv = numerical_curvature(points, ub) + D
    # Numerical integral using the composite trapezoidal rule
    F = cumtrapz(cv, ub, initial=0)
    F /= F[-1]
    func = interp1d(F, ub)
    ff = np.linspace(0, 1, num=points.shape[0])
    ub = func(ff)
    return ub

def synthesize(x, U_upper, U_lower, p, n_points, return_cp=False):
    n_vars = len(x)
    x_upper = x[:n_vars/2]
    x_lower = x[n_vars/2:]
    cp_upper = x2ctrlpts_single_curve(x_upper)
    cp_lower = x2ctrlpts_single_curve(x_lower)
    n_control_points = cp_upper.shape[0]
    ub = np.linspace(0.0, 1.0, n_points/2)
    weights = np.ones(n_control_points)
    points_upper = eval_curve(cp_upper, weights, U_upper, p, ub)
    points_lower = eval_curve(cp_lower, weights, U_lower, p, ub)
    # Rearrange points along the curves
    D = 5.0
    ub_upper = new_ub_single_curve(points_upper, ub, D)
    ub_lower = new_ub_single_curve(points_lower, ub, D)
    points_upper = eval_curve(cp_upper, weights, U_upper, p, ub_upper)
    points_lower = eval_curve(cp_lower, weights, U_lower, p, ub_lower)
    # Concatenate upper and lower curves
    airfoil = np.vstack((np.flip(points_upper, axis=0), points_lower))
    if return_cp:
        return airfoil, cp_upper, cp_lower
    else:
        return airfoil


    
