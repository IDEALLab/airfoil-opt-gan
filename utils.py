"""
Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import itertools
import time

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def optimize_kde(X):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-3, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, n_jobs=8, cv=5, verbose=1)
    grid.fit(X)
    
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    
    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    
    return kde

def convert_sec(sec):
    if sec < 60:
        return "%.2f sec" % sec
    elif sec < (60 * 60):
        return "%.2f min" % (sec / 60)
    else:
        return "%.2f hr" % (sec / (60 * 60))

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed_time(self):
        return convert_sec(time.time() - self.start_time)
    
def gen_grid(d, points_per_axis, lb=0., rb=1.):
    ''' Generate a grid in a d-dimensional space 
        within the range [lb, rb] for each axis '''
    
    lincoords = []
    for i in range(0, d):
        lincoords.append(np.linspace(lb, rb, points_per_axis))
    coords = list(itertools.product(*lincoords))
    
    return np.array(coords)

def mean_err(metric_list):
    n = len(metric_list)
    mean = np.mean(metric_list)
    std = np.std(metric_list)
    err = 1.96*std/n**.5
    return mean, err

def visualize(X):
    
    X = X.reshape((X.shape[0], -1))
    pca = PCA(n_components=3)
    F = pca.fit_transform(X)
    
    # Reconstruction error
    X_rec = pca.inverse_transform(F)
    err = mean_squared_error(X, X_rec)
    print 'Reconstruct error: ', err
    
    # 3D Plot
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection = '3d')
    
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([F[:,0].max()-F[:,0].min(), F[:,1].max()-F[:,1].min(), F[:,2].max()-F[:,2].min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(F[:,0].max()+F[:,0].min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(F[:,1].max()+F[:,1].min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(F[:,2].max()+F[:,2].min())
    ax3d.scatter(Xb, Yb, Zb, c='white', alpha=0)
    
    ax3d.scatter(F[:,0], F[:,1], F[:,2])
    matplotlib.rcParams.update({'font.size': 22})
#    ax3d.set_xticks([])
#    ax3d.set_yticks([])
#    ax3d.set_zticks([])
    plt.show()
    
def safe_remove(filename):
    if os.path.exists(filename):
        os.remove(filename)
