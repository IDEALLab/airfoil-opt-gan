# Generate and plot the contour of an airfoil 
# using the PARSEC parameterization

# Repository & documentation:
# http://github.com/dqsis/parsec-airfoils
# -------------------------------------


# Import libraries
from __future__ import division
from math import sqrt, tan, pi
import numpy as np

# User function pcoef
def pcoef(xte, yte, rle, x_cre, y_cre, d2ydx2_cre, th_cre, surface):
    
    # Docstrings
    """evaluate the PARSEC coefficients"""

    # Initialize coefficients
    coef = np.zeros(6)

    # 1st coefficient depends on surface (pressure or suction)
    if surface.startswith('p'):
        coef[0] = -sqrt(2*rle)
    else:
        coef[0] = sqrt(2*rle)
 
    # Form system of equations
    A = np.array([
                 [xte**1.5, xte**2.5, xte**3.5, xte**4.5, xte**5.5],
                 [x_cre**1.5, x_cre**2.5, x_cre**3.5, x_cre**4.5, 
                  x_cre**5.5],
                 [1.5*sqrt(xte), 2.5*xte**1.5, 3.5*xte**2.5, 
                  4.5*xte**3.5, 5.5*xte**4.5],
                 [1.5*sqrt(x_cre), 2.5*x_cre**1.5, 3.5*x_cre**2.5, 
                  4.5*x_cre**3.5, 5.5*x_cre**4.5],
                 [0.75*(1/sqrt(x_cre)), 3.75*sqrt(x_cre), 8.75*x_cre**1.5, 
                  15.75*x_cre**2.5, 24.75*x_cre**3.5]
                 ]) 

    B = np.array([
                 [yte - coef[0]*sqrt(xte)],
                 [y_cre - coef[0]*sqrt(x_cre)],
                 [tan(th_cre*pi/180) - 0.5*coef[0]*(1/sqrt(xte))],
                 [-0.5*coef[0]*(1/sqrt(x_cre))],
                 [d2ydx2_cre + 0.25*coef[0]*x_cre**(-1.5)]
                 ])
    
    # Solve system of linear equations
#    X = np.linalg.solve(A,B)
    X = np.linalg.lstsq(A,B)[0]

    # Gather all coefficients
    coef[1:6] = X[0:5,0]

    # Return coefficients
    return coef

def synthesize(parsec_params, n_points=100):
    
    # TE & LE of airfoil (normalized, chord = 1)
    xle = 0.0
    yle = 0.0
    xte = 1.0
    yte = 0.0
    
    # LE radius
    rle = parsec_params[0]

    # Pressure (lower) surface parameters 
    x_pre = parsec_params[1]
    y_pre = parsec_params[2]
    d2ydx2_pre = parsec_params[3]
    th_pre = parsec_params[4]
    
    # Suction (upper) surface parameters
    x_suc = parsec_params[5]
    y_suc = parsec_params[6]
    d2ydx2_suc = parsec_params[7]
    th_suc = parsec_params[8]
    
    # Evaluate pressure (lower) surface coefficients
    cf_pre = pcoef(xte, yte, rle, x_pre, y_pre, d2ydx2_pre, th_pre, 'pre')
    
    # Evaluate suction (upper) surface coefficients
    cf_suc = pcoef(xte, yte, rle, x_suc, y_suc, d2ydx2_suc, th_suc, 'suc')
    
    if n_points%2 == 0:
        n_points_pre = n_points_suc = n_points/2
    else:
        n_points_pre = n_points/2+1
        n_points_suc = n_points/2
    
    # Evaluate pressure (lower) surface points
    xx_pre = np.linspace(xte, xle, n_points_pre)
    yy_pre = (cf_pre[0]*xx_pre**(1/2) + 
              cf_pre[1]*xx_pre**(3/2) + 
              cf_pre[2]*xx_pre**(5/2) + 
              cf_pre[3]*xx_pre**(7/2) + 
              cf_pre[4]*xx_pre**(9/2) + 
              cf_pre[5]*xx_pre**(11/2)
             ) 
    
    # Evaluate suction (upper) surface points
    xx_suc = np.linspace(xle, xte, n_points_suc)
    yy_suc = (cf_suc[0]*xx_suc**(1/2) + 
              cf_suc[1]*xx_suc**(3/2) + 
              cf_suc[2]*xx_suc**(5/2) + 
              cf_suc[3]*xx_suc**(7/2) + 
              cf_suc[4]*xx_suc**(9/2) + 
              cf_suc[5]*xx_suc**(11/2)
             )
    
    xy_pre = np.vstack((xx_pre, yy_pre)).T
    xy_suc = np.vstack((xx_suc, yy_suc)).T
    x = np.vstack((xy_pre, xy_suc))
    x = np.flip(x, 0)
    
    return x
