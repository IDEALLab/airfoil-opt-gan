"""
Weighted and constrained B-spline approximation.

Author(s): Wei Chen (wchen459@umd.edu)

Reference: Piegl, L., & Tiller, W. (2012). The NURBS book. Springer Science & Business Media.

n+1 : number of control points
r+1 : number of data points
"""


import numpy as np
from bspline import *


def wc_least_squares_curve(Q, Wq, D, I, Wd, n, p):
    ''' Weighted & constrained least squares curve fit. '''
    
    r = Q.shape[0]-1
    s = D.shape[0]-1
    
    ru = rc = -1
    for i in range(r+1):
        if Wq[i] > 0:
            ru += 1
        else:
            rc += 1
    su = sc = -1
    for j in range(s+1):
        if Wd[j] > 0:
            su += 1
        else:
            sc += 1
    mu = ru + su + 1
    mc = rc + sc + 1
    assert mc < n and mc+n < mu+1
    ub = choose_ub(Q)
    U = choose_knots(ub, n, p, mode='deboor') # initial U
    # Set up arrays N, W, S, T, M
    N = np.zeros((mu+1, n+1))
    M = np.zeros((mc+1, n+1))
    S = np.zeros((mu+1, 2))
    T = np.zeros((mc+1, 2))
    A = np.zeros(mc+1)
    W = np.zeros(mu+1)
    funs = np.zeros((2, n+1))
    j = 0 # current index into I
    mu2 = 0
    mc2 = 0 # counters up to mu and mc
    for i in range(r+1):
        span = find_span(n, p, ub[i], U)
        dflag = 0
        if j <= s and i == I[j]:
            dflag = 1
        if dflag == 0:
            funs[0] = basis_funs(span, ub[i], n, p, U)
        else:
            funs[1] = ders_basis_funs(span, ub[i], n, p, 1, U)[1]
        if Wq[i] > 0.0:
            # Uconstrained point
            W[mu2] = Wq[i]
            N[mu2] = funs[0]
            S[mu2] = W[mu2]*Q[i]
            mu2 += 1
        else:
            # Constrained point
            M[mc2] = funs[0]
            T[mc2] = Q[i]
            mc2 += 1
        if dflag == 1:
            # Derivative at this point
            if Wd[j] > 0.0:
                # Unconstrained derivative
                W[mu2] = Wd[j]
                N[mu2] = funs[1]
                S[mu2] = W[mu2]*D[j]
                mu2 += 1
            else:
                # Constrained derivative
                M[mc2] = funs[1]
                T[mc2] = D[j]
                mc2 += 1
            j += 1
    W = np.diag(W)
    NWN = np.dot(np.dot(N.T, W), N)
    NWS = np.dot(N.T, S)
    if mc < 0:
        # No constraints
        P = np.linalg.solve(NWN, NWS)
        return P, U
    NWN_inv = np.linalg.pinv(NWN)
    a = np.dot(np.dot(M, NWN_inv), M.T)
    b = np.dot(np.dot(np.dot(M, NWN_inv), N.T), S) - T
#    A = np.linalg.solve(a, b)
    A = np.linalg.lstsq(a, b)[0]
    P = np.dot(NWN_inv, NWS-np.dot(M.T, A))
    return P, U

