"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from utils import gen_grid


def plot_shape(xys, z1, z2, ax, scale, scatter, symm_axis, **kwargs):
#    mx = max([y for (x, y) in m])
#    mn = min([y for (x, y) in m])
    xscl = scale# / (mx - mn)
    yscl = scale# / (mx - mn)
#    ax.scatter(z1, z2)
    if scatter:
        if 'c' not in kwargs:
            kwargs['c'] = cm.rainbow(np.linspace(0,1,xys.shape[0]))
#        ax.plot( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        ax.scatter( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), edgecolors='none', **kwargs)
    else:
        ax.plot( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), **kwargs)
        
    if symm_axis == 'y':
#        ax.plot( *zip(*[(-x * xscl + z1, y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        plt.fill_betweenx( *zip(*[(y * yscl + z2, -x * xscl + z1, x * xscl + z1)
                          for (x, y) in xys]), color='gray', alpha=.2)
    elif symm_axis == 'x':
#        ax.plot( *zip(*[(x * xscl + z1, -y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        plt.fill_between( *zip(*[(x * xscl + z1, -y * yscl + z2, y * yscl + z2)
                          for (x, y) in xys]), color='gray', alpha=.2)

def plot_samples(Z, X, P=None, W=None, scale=0.8, scatter=True, symm_axis=None, annotate=False, fname=None, **kwargs):
    
    ''' Plot shapes given design sapce and latent space coordinates '''
    
    plt.rc("font", size=12)
    
    if Z is None or Z.shape[1] != 2:
        N = X.shape[0]
        points_per_axis = int(N**.5)
        bounds = (-1., 1.)
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
    
    r0 = np.max(Z[:,0]) - np.min(Z[:,0])
    r1 = np.max(Z[:,1]) - np.min(Z[:,1])
    scale *= np.minimum(r0, r1)/points_per_axis
        
    # Create a 2D plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
            
    for (i, z) in enumerate(Z):
        if P is not None and W is not None:
            ind = W[i] >= 0
            p = P[i][ind]
            w = W[i][ind]
            w /= np.max(w)
            plot_shape(p, z[0], .3*z[1], ax, scale, scatter, symm_axis, c='g', s=10*w, marker='P', alpha=.5)
            plot_shape(p, z[0], .3*z[1], ax, scale, False, symm_axis, lw=.5, alpha=.5, c='g')
        plot_shape(X[i], z[0], .3*z[1], ax, scale, scatter, symm_axis, **kwargs)
        if annotate:
            label = '{0}'.format(i+1)
            plt.annotate(label, xy = (z[0], z[1]), size=10)
    
#    plt.xlabel('c1')
#    plt.ylabel('c2')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(fname+'.svg', dpi=600)
    plt.close()

def plot_synthesized(Z, gen_func, d=2, scale=.8, scatter=True, symm_axis=None, fname=None, **kwargs):
    
    ''' Synthesize shapes given latent space coordinates and plot them '''
    
    if d == 2:
        latent = Z
    else:
        latent = np.random.normal(scale=0.5, size=(Z.shape[0], d))
    X = gen_func(latent)
    P = W = None
    
    plot_samples(Z, X, P, W, scale, scatter, symm_axis, fname=fname, **kwargs)

def plot_grid(points_per_axis, gen_func, d=2, bounds=(0.0, 1.0), scale=.8, scatter=True, 
              symm_axis=None, fname=None, **kwargs):
    
    ''' Uniformly plots synthesized shapes in the latent space
        K : number of samples for each point in the latent space '''
    
    if d == 1:
        Z = np.linspace(bounds[0], bounds[1], points_per_axis)
        Z = np.vstack((Z, np.zeros(points_per_axis))).T
        plot_synthesized(Z, gen_func, 2, scale, scatter, symm_axis, fname, **kwargs)
    if d == 2:
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
        plot_synthesized(Z, gen_func, 2, scale, scatter, symm_axis, fname, **kwargs)
    if d >= 3:
        Z = np.random.uniform(bounds[0], bounds[1], (points_per_axis**2, d))
        zgrid = np.linspace(bounds[0], bounds[1], points_per_axis)
        for i in range(points_per_axis):
            Zxy = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
            Zz = np.ones((points_per_axis**2, 1)) * zgrid[i]
            Z[:, :3] = np.hstack((Zxy, Zz)) # random after 3rd dimension
            plot_synthesized(Z, gen_func, 2, scale, scatter, symm_axis, '%s_%.2f' % (fname, zgrid[i]), **kwargs)
        

