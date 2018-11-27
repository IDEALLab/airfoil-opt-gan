
"""
Optimize the airfoil shape in the latent space using Bayesian optimization, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.gaussian_process as gp

from simulation import evaluate
from bayesian_opt import normalize, neg_expected_improvement, sample_next_point
from shape_plot import plot_samples
from utils import mean_err


def gen_func(z, pca):
    airfoil = pca.inverse_transform(z)
    airfoil = airfoil.reshape((airfoil.shape[0], -1, 2))
    return airfoil

def synthesize(z, pca):
    airfoil = gen_func(z, pca)
    airfoil = np.squeeze(airfoil)
    if airfoil[0,1] < airfoil[-1,1]:
        mean = .5*(airfoil[0,1]+airfoil[-1,1])
        airfoil[0,1] = airfoil[-1,1] = mean
    return airfoil

def optimize(latent_dim, bounds, n_eval, run_id):
    # Optimize in the latent space
    n_pre_samples = 10
    kernel = gp.kernels.Matern()
    gp_model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=100, normalize_y=False)
    zp = []
    perfs = []
    opt_perfs = [0]
    s = 1.0
    gamma = 0.99
    for i in range(n_eval):
        if len(zp) < n_pre_samples:
            z = np.random.uniform(bounds[:,0], bounds[:,1], size=latent_dim)
        else:
            perf_normalized = normalize(perfs)
            gp_model.fit(np.array(zp), np.array(perf_normalized))
            length_scale = gp_model.kernel_.length_scale
            print('Length scale = {}'.format(length_scale))
            previous_optimum = perf_normalized[opt_idx]
            if np.all(np.array(perfs[-5:])==-1): # in case getting stuck in infeasible region
                previous_point = opt_z
            else:
                previous_point = z
#            z = sample_next_point(latent_dim, neg_expected_improvement, gp_model, previous_optimum, bounds, n_restarts=100)
            z = sample_next_point(latent_dim, neg_expected_improvement, gp_model, previous_optimum, bounds=None, 
                                  random_search=100000, previous_point=previous_point, scale=s)
            s *= gamma
            
        x = synthesize(z.reshape(1,-1), pca)
        perf = evaluate(x)
        z = np.squeeze(z)
        zp.append(z)
        perfs.append(perf)
        opt_idx = np.argmax(perfs)
        opt_z = zp[opt_idx]
        opt_perf = perfs[opt_idx]
        opt_perfs.append(opt_perf) # Best performance so far
        print('PCA-BO {}-{}: z {} CL/CD {:.2f} best-so-far {:.2f}'.format(run_id, i+1, z, perf, opt_perf))
        
    opt_z = opt_z.reshape(1,-1)
    opt_airfoil = synthesize(opt_z, pca)
    print('Optimal: z {} CL/CD {}'.format(opt_z, opt_perfs[-1]))
    
    return opt_airfoil, opt_perfs


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Optimize')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--n_eval', type=int, default=1000, help='number of evaluations per run')
    args = parser.parse_args()
    
    n_runs = args.n_runs
    n_eval = args.n_eval
    
    # Read data
    data_fname = 'airfoil_interp.npy'
    X = np.load(data_fname)
    # Split training and test data
    test_split = 0.8
    N = X.shape[0]
    split = int(N*test_split)
    X_train = X[:split]
    X_test = X[split:]
    
    # Draw a scree plot
#    d_max = 10
#    X = np.reshape(X, (X.shape[0], -1))
#    pca = PCA(d_max).fit(X)
#    evrs = pca.explained_variance_ratio_ # explained variance ratios
#    plt.figure()
#    dims = np.arange(1, d_max+1)
#    plt.plot(dims, evrs, 'ko-')
#    plt.xlabel('Principle components')
#    plt.ylabel('Explained variance ratio')
#    plt.title('Scree Plot')
#    plt.savefig('pca/scree.svg')
#    plt.close()
    
    # Fit a PCA model
    latent_dim = 3
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    pca = PCA(latent_dim).fit(X_train)
    z = pca.transform(X_train)
    bounds = np.zeros((latent_dim, 2))
    bounds[:,0] = np.min(z, axis=0)
    bounds[:,1] = np.max(z, axis=0)
    
    # Visualize the latent space
#    bounds = np.zeros((latent_dim, 2))
#    bounds[:,0] = np.min(z, axis=0)
#    bounds[:,1] = np.max(z, axis=0)
#    points_per_axis = 5
#    xgrid = np.linspace(bounds[0,0], bounds[0,1], points_per_axis)
#    ygrid = np.linspace(bounds[1,0], bounds[1,1], points_per_axis)
#    zgrid = np.linspace(bounds[2,0], bounds[2,1], points_per_axis)
#    Zx, Zy = np.meshgrid(xgrid, ygrid) # Generate a grid
#    Zxy = np.hstack((Zx.reshape(-1,1), Zy.reshape(-1,1)))
#    for i in range(points_per_axis):
#        Zz = np.ones((points_per_axis**2, 1)) * zgrid[i]
#        Z = np.hstack((Zxy, Zz))
#        X = gen_func(Z, pca)
#        plot_samples(Z, X, scale=1.0, scatter=False, fname='pca/synthesized_%.2f' % zgrid[i], alpha=.7, lw=1.2, c='k')
    
    # Airfoil parameters
    n_points = 192
    
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    for i in range(n_runs):
        start_time = time.time()
        opt_airfoil, opt_perfs = optimize(latent_dim, bounds, n_eval, i+1)
        end_time = time.time()
        opt_airfoil_runs.append(opt_airfoil)
        opt_perfs_runs.append(opt_perfs)
        time_runs.append(end_time-start_time)
    
    opt_airfoil_runs = np.array(opt_airfoil_runs)
    opt_perfs_runs = np.array(opt_perfs_runs)
    np.save('opt_results/pca_bo/opt_airfoil.npy', opt_airfoil_runs)
    np.save('opt_results/pca_bo/opt_history.npy', opt_perfs_runs)
    
    # Plot optimization history
    mean_perfs_runs = np.mean(opt_perfs_runs, axis=0)
    plt.figure()
    plt.plot(np.arange(n_eval+1, dtype=int), mean_perfs_runs)
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Optimal CL/CD')
#    plt.xticks(np.linspace(0, n_eval+1, 5, dtype=int))
    plt.savefig('opt_results/pca_bo/opt_history.svg')
    plt.close()
    
    # Plot the optimal airfoil
    mean_time_runs, err_time_runs = mean_err(time_runs)
    mean_final_perf_runs, err_final_perf_runs = mean_err(opt_perfs_runs[:,-1])
    plt.figure()
    for opt_airfoil in opt_airfoil_runs:
        plt.plot(opt_airfoil[:,0], opt_airfoil[:,1], '-', c='k', alpha=1.0/n_runs)
    plt.title('CL/CD: %.2f+/-%.2f  time: %.2f+/-%.2f min' % (mean_final_perf_runs, err_final_perf_runs, 
                                                             mean_time_runs/60, err_time_runs/60))
    plt.axis('equal')
    plt.savefig('opt_results/pca_bo/opt_airfoil.svg')
    plt.close()

    print 'PCA-BO completed :)'
