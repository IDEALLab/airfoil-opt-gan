"""
B-spline algorithms.

Author(s): Wei Chen (wchen459@umd.edu)

Reference: Piegl, L., & Tiller, W. (2012). The NURBS book. Springer Science & Business Media.

n+1 : number of control points
n+p+2 : number of knots
n-p+1 : number of internal knot spans
m+1 : number of data points
"""

import numpy as np
from matplotlib import pyplot as plt


EPSILON = np.finfo(np.float32).eps

def find_span(n, p, u, U):
    ''' Determine the knot span index. '''
    if u == U[n+1]:
        # Special case
        return n
    low = p
    high = n+1
    # Binary search
    mid = (low+high)/2
    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low+high)/2
    return mid

def basis_funs(span, u, n, p, U, full=True):
    ''' Compute the nonvanishing basis functions. '''
    left = np.zeros(p+1)
    right = np.zeros(p+1)
    N = np.zeros(p+1)
    N[0] = 1.0
    for j in range(1, p+1):
        left[j] = u - U[span+1-j]
        right[j] = U[span+j] - u
        saved = 0.0
        for r in range(j):
            temp = N[r]/(right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        N[j] = saved
    if not full:
        return N
    temp = N
    N = np.zeros(n+1)
    N[span-p:span+1] = temp
    return N

def ders_basis_funs(span, u, n, p, l, U, full=True):
    ''' Compute nonzero basis functions and their derivatives. '''
    left = np.zeros(p+1)
    right = np.zeros(p+1)
    ndu = np.ones((p+1, p+1))
    for j in range(1, p+1):
        left[j] = u - U[span+1-j]
        right[j] = U[span+j] - u
        saved = 0.0
        for r in range(j):
            # Lower triangle
            ndu[j,r] = right[r+1] + left[j-r]
            temp = ndu[r,j-1]/ndu[j,r]
            # Upper triangle
            ndu[r,j] = saved + right[r+1]*temp
            saved = left[j-r]*temp
        ndu[j,j] = saved
    ders = np.zeros((l+1,p+1))
    # Load the basis functions
    for j in range(p+1):
        ders[0,j] = ndu[j,p]
    # Compute the derivatives (Eq. 2.9)
    a = np.ones((p+1, p+1))
    for r in range(p+1):
        s1 = 0
        s2 = 1
        # Loop to compute kth derivative
        for k in range(1, l+1):
            d = 0.0
            rk = r-k
            pk = p-k
            if r >= k:
                a[s2,0] = a[s1,0]/ndu[pk+1,rk]
                d = a[s2,0]*ndu[rk,pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r-1 <=pk:
                j2 = k-1
            else:
                j2 = p-r
            for j in range(j1, j2+1):
                a[s2,j] = (a[s1,j]-a[s1,j-1])/ndu[pk+1,rk+j]
                d += a[s2,j]*ndu[rk+j,pk]
            if r<= pk:
                a[s2,k] = -a[s1,k-1]/ndu[pk+1,r]
                d += a[s2,k]*ndu[r,pk]
            ders[k,r] = d
            # Switch rows
            j = s1
            s1 = s2
            s2 = j
    # Multiply through by the correct factors (Eq. 2.9)
    r = p
    for k in range(1, l+1):
        for j in range(p+1):
            ders[k,j] *= r
        r *= (p-k)
    if not full:
        return ders
    temp = ders
    ders = np.zeros((l+1,n+1))
    ders[:, span-p:span+1] = temp
    return ders

def choose_ub(Q):
    ''' Eq. 9.5 '''
    D = np.diff(Q, axis=0)
    d = np.linalg.norm(D, axis=1)
    s = np.sum(d)
    m = Q.shape[0]-1
    ub = np.zeros(m+1)
    ub[1:] = d/s
    ub = np.cumsum(ub)
    ub[-1] = 1.0
    return ub

def choose_knots(ub, n, p, mode='uniform'):
    assert mode in ['uniform', 'averaging', 'deboor']
    U = np.zeros(n+p+2)
    # Uniform knots
    if mode == 'uniform':
        U[p:-p] = np.linspace(0, 1, n-p+2)
    if mode == 'averaging':
        for j in range(1, n-p+1):
            U[j+p] = np.mean(ub[j:j+p])
    if mode == 'deboor':
        # Eqs. 9.68 and 9.69
        m = len(ub)-1
        d = float(m+1)/(n-p+1)
        for j in range(1, n-p+1):
            i = int(j*d)
            alpha = j*d - i
            U[p+j] = (1-alpha)*ub[i-1] + alpha*ub[i]
    U[-p-1:] = 1
    return U
    
def curve_point(u, P, W, U, p):
    ''' Compute point on rational B-spline curve. '''
    n = P.shape[0] - 1
    Pw = np.zeros((n+1, 3))
    Pw[:,:2] = W.reshape((-1,1))*P
    Pw[:,2] = W
    span = find_span(n, p, u, U)
    N = basis_funs(span, u, n, p, U)
    Cw = np.sum(N.reshape((-1,1)) * Pw, axis=0)
    C = Cw[:2]/Cw[2]
    return C

def eval_knots(P, W, U, p):
    n_knots = len(U)
    V = np.zeros((n_knots, 2))
    for i in range(n_knots):
        V[i] = curve_point(U[i], P, W, U, p)
    return V

def eval_curve(P, W, U, p, u):
    C = [curve_point(u[i], P, W, U, p) for i in range(len(u))]
    return np.array(C)

def eval_error(Q, P, W, U, p, mode='max'):
    ub = choose_ub(Q)
    C = eval_curve(P, W, U, p, ub)
    error = np.linalg.norm(C - Q, axis=1)
    assert mode in ['max', 'mean', 'full']
    if mode == 'mean':
        return np.mean(error)
    elif mode == 'max':
        return np.max(error)
    elif mode == 'full':
        return error
    
def numerical_curvature(Q, ub):
    ''' Estimate curvature at each data point using central difference. '''
    dx_du = np.gradient(Q[:,0], ub)
    dy_du = np.gradient(Q[:,1], ub)
    d2x_du2 = np.gradient(dx_du, ub)
    d2y_du2 = np.gradient(dy_du, ub)
    cv = np.abs(d2x_du2*dy_du - dx_du*d2y_du2)/(dx_du*dx_du + dy_du*dy_du)**1.5
    return cv

def plot_curve(Q, P, W, U, p, n_points=200):
    u = np.linspace(0, 1, n_points)
    C = eval_curve(P, W, U, p, u)
    plt.figure()
    plt.plot(C[:,0], C[:,1], 'b-', linewidth=2, alpha=0.5)
    plt.plot(Q[:,0], Q[:,1], 'r-', alpha=0.5)
    plt.plot(P[:,0], P[:,1], 'go-', alpha=0.3)
    V = eval_knots(P, W, U, p)
    plt.plot(V[:,0], V[:,1], 'ko', alpha=0.5)
    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    err = eval_error(Q, P, W, U, p)
    plt.title('error: %f ' % err)
    plt.show()

    