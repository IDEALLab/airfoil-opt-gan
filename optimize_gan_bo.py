
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
from importlib import import_module
import sklearn.gaussian_process as gp

from gan import GAN
from simulation import evaluate
from bayesian_opt import normalize, neg_expected_improvement, sample_next_point
from utils import mean_err


def synthesize(z, model):
    airfoil = model.synthesize(z)
    if airfoil[0,1] < airfoil[-1,1]:
        mean = .5*(airfoil[0,1]+airfoil[-1,1])
        airfoil[0,1] = airfoil[-1,1] = mean
    return airfoil

def optimize(latent_dim, bounds, n_eval, run_id):
    # Optimize in the latent space
    n_pre_samples = 10
    bounds = np.tile(bounds, (latent_dim,1))
    kernel = gp.kernels.Matern()
    gp_model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=100, normalize_y=False)
    zp = []
    perfs = []
    opt_perfs = [0]
    s = 1.0
    gamma = 0.99
    for i in range(n_eval):
        if i < n_pre_samples:
            z = np.random.uniform(bounds[:,0], bounds[:,1], size=latent_dim)
        else:
            perf_normalized = normalize(perfs)
            gp_model.fit(np.array(zp), np.array(perf_normalized))
            length_scale = gp_model.kernel_.length_scale
            print('Length scale = {}'.format(length_scale))
            previous_optimum = perf_normalized[opt_idx]
            if np.all(np.array(perfs[-5:])==-1): # in case getting stuck in infeasible region
                print('Back to {} ...'.format(opt_z))
                previous_point = opt_z
            else:
                previous_point = z
#            z = sample_next_point(latent_dim, neg_expected_improvement, gp_model, previous_optimum, bounds, n_restarts=100)
            z = sample_next_point(latent_dim, neg_expected_improvement, gp_model, previous_optimum, bounds=None, 
                                  random_search=100000, previous_point=previous_point, scale=s)
            s *= gamma
            
        x = synthesize(z.reshape(1,-1), model)
        perf = evaluate(x)
        z = np.squeeze(z)
        zp.append(z)
        perfs.append(perf)
        opt_idx = np.argmax(perfs)
        opt_z = zp[opt_idx]
        opt_perf = perfs[opt_idx]
        opt_perfs.append(opt_perf) # Best performance so far
        print('GAN-BO {}-{}: z {} CL/CD {:.2f} best-so-far {:.2f}'.format(run_id, i+1, z, perf, opt_perf))
        
    opt_z = opt_z.reshape(1,-1)
    opt_airfoil = synthesize(opt_z, model)
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
    
    # Airfoil parameters
    latent_dim = 3
    noise_dim = 10
    n_points = 192
    bezier_degree = 32
    bounds = (0., 1.)
    
    # Restore trained model
    model = GAN(latent_dim, noise_dim, n_points, bezier_degree, bounds)
    model.restore()
    
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
    np.save('opt_results/gan_bo/opt_airfoil.npy', opt_airfoil_runs)
    np.save('opt_results/gan_bo/opt_history.npy', opt_perfs_runs)
    
    # Plot optimization history
    mean_perfs_runs = np.mean(opt_perfs_runs, axis=0)
    plt.figure()
    plt.plot(np.arange(n_eval+1, dtype=int), mean_perfs_runs)
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Optimal CL/CD')
#    plt.xticks(np.linspace(0, n_eval+1, 5, dtype=int))
    plt.savefig('opt_results/gan_bo/opt_history.svg')
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
    plt.savefig('opt_results/gan_bo/opt_airfoil.svg')
    plt.close()

    print 'GAN-BO completed :)'
