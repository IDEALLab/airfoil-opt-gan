"""
Optimize the airfoil shape directly using genetic algorithm, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)

Reference(s):
    Viswanath, A., J. Forrester, A. I., Keane, A. J. (2011). Dimension Reduction for Aerodynamic Design Optimization.
    AIAA Journal, 49(6), 1256-1266.
    Grey, Z. J., Constantine, P. G. (2018). Active subspaces of airfoil shape parameterizations.
    AIAA Journal, 56(5), 2003-2017.
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from nurbs.synthesis import synthesize
from nurbs.fitting import nurbs_airfoil
from genetic_alg import generate_first_population, select, create_children, mutate_population
from utils import mean_err


def optimize(x0, syn_func, perturb_type, perturb, n_eval, run_id):
    # Optimize using GA
    n_best = 30
    n_random = 10
    n_children = 5
    chance_of_mutation = 0.1
    population_size = int((n_best+n_random)/2*n_children)
    population = generate_first_population(x0, population_size, perturb_type, perturb)
    best_inds = []
    best_perfs = []
    opt_perfs = [0]
    i = 0
    while 1:
        breeders, best_perf, best_individual = select(population, n_best, n_random, syn_func)
        best_inds.append(best_individual)
        best_perfs.append(best_perf)
        opt_perfs += [np.max(best_perfs)] * population_size # Best performance so far
        print('NURBS-GA %d-%d: fittest %.2f' % (run_id, i+1, best_perf))
        # No need to create next generation for the last generation
        if i < n_eval/population_size-1:
            next_generation = create_children(breeders, n_children)
            population = mutate_population(next_generation, chance_of_mutation, perturb_type, perturb)
            i += 1
        else:
            break
    
    opt_x = best_inds[np.argmax(best_perfs)]
    opt_airfoil = syn_func(opt_x)
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
    n_control_points = 5 # for each curve (upper or lower)
    p = 3
    
    # NACA 0012 as the original airfoil
    data_path = './initial_airfoil/naca0012.dat'
    x0, U_upper, U_lower = nurbs_airfoil(n_control_points, p, data_path)
    syn_func = lambda x: synthesize(x, U_upper, U_lower, p, n_points)
    
    perturb_type = 'absolute'
    perturb = 0.1
    
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    for i in range(n_runs):
        start_time = time.time()
        opt_airfoil, opt_perfs = optimize(x0, syn_func, perturb_type, perturb, n_eval, i+1)
        end_time = time.time()
        opt_airfoil_runs.append(opt_airfoil)
        opt_perfs_runs.append(opt_perfs)
        time_runs.append(end_time-start_time)
    
    opt_airfoil_runs = np.array(opt_airfoil_runs)
    opt_perfs_runs = np.array(opt_perfs_runs)
    np.save('opt_results/nurbs_ga/opt_airfoil.npy', opt_airfoil_runs)
    np.save('opt_results/nurbs_ga/opt_history.npy', opt_perfs_runs)
    
    # Plot optimization history
    mean_perfs_runs = np.mean(opt_perfs_runs, axis=0)
    plt.figure()
    plt.plot(np.arange(n_eval+1, dtype=int), opt_perfs)
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Optimal CL/CD')
#    plt.xticks(np.linspace(0, n_eval+1, 5, dtype=int))
    plt.savefig('opt_results/nurbs_ga/opt_history.svg')
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
    plt.savefig('opt_results/nurbs_ga/opt_airfoil.svg')
    plt.close()

    print 'NURBS-GA completed :)'
