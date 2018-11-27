"""
Optimize the airfoil shape directly using Bayesian optimization, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp

from simulation import evaluate
from bayesian_opt import normalize, neg_expected_improvement, sample_next_point
from parsec.synthesis import synthesize
from utils import mean_err


def optimize(bounds, n_eval, run_id):
    # Optimize using BO
    n_pre_samples = 10
    kernel = gp.kernels.Matern()
    gp_model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=100, normalize_y=False)
    xs = []
    perfs = []
    opt_perfs = [0]
    for i in range(n_eval):
        if len(xs) < n_pre_samples:
            x = np.random.uniform(bounds[:,0], bounds[:,1], size=d)
        else:
            perfs_normalized = normalize(perfs)
            gp_model.fit(np.array(xs), np.array(perfs_normalized))
            print('Length scale = {}'.format(gp_model.kernel_.length_scale))
            previous_optimum = np.max(perfs_normalized)
#            x = sample_next_point(d, neg_expected_improvement, gp_model, previous_optimum, bounds, n_restarts=100)
            x = sample_next_point(d, neg_expected_improvement, gp_model, previous_optimum, bounds, random_search=100000)
        airfoil = synthesize(x, n_points)
        perf = evaluate(airfoil)
        print('PARSEC-BO %d-%d: CL/CD %.2f' % (run_id, i+1, perf))
        xs.append(x)
        perfs.append(perf)
        opt_perfs.append(np.max(perfs)) # Best performance so far
    
    opt_x = xs[np.argmax(perfs)]
    opt_airfoil = synthesize(opt_x, n_points)
    print('Optimal CL/CD: {}'.format(opt_perfs[-1]))
    
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
    n_points = 192
    
    # NACA 0012 as the original airfoil
    x0 = np.array([0.0147, 0.2996, -0.06, 0.4406, 7.335, 0.3015, 0.0599, -0.4360, -7.335]) # NACA 0012
    d = len(x0)
    bounds = np.zeros((d, 2))
    perturb = 0.2
    bounds[:,0] = x0 * (1-perturb)
    bounds[:,1] = x0 * (1+perturb)
    
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    for i in range(n_runs):
        start_time = time.time()
        opt_airfoil, opt_perfs = optimize(bounds, n_eval, i+1)
        end_time = time.time()
        opt_airfoil_runs.append(opt_airfoil)
        opt_perfs_runs.append(opt_perfs)
        time_runs.append(end_time-start_time)
    
    opt_airfoil_runs = np.array(opt_airfoil_runs)
    opt_perfs_runs = np.array(opt_perfs_runs)
    np.save('opt_results/parsec_bo/opt_airfoil.npy', opt_airfoil_runs)
    np.save('opt_results/parsec_bo/opt_history.npy', opt_perfs_runs)
    
    # Plot optimization history
    mean_perfs_runs = np.mean(opt_perfs_runs, axis=0)
    plt.figure()
    plt.plot(np.arange(n_eval+1, dtype=int), mean_perfs_runs)
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Optimal CL/CD')
#    plt.xticks(np.linspace(0, n_eval+1, 5, dtype=int))
    plt.savefig('opt_results/parsec_bo/opt_history.svg')
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
    plt.savefig('opt_results/parsec_bo/opt_airfoil.svg')
    plt.close()

    print 'PARSEC-BO completed :)'
