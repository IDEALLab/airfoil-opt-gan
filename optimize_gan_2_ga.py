
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
from genetic_alg import generate_first_population, select, create_children, mutate_population
from utils import mean_err


def synthesize(z, model, noise=None):
    airfoil = model.synthesize(z, noise)
    if airfoil[0,1] < airfoil[-1,1]:
        mean = .5*(airfoil[0,1]+airfoil[-1,1])
        airfoil[0,1] = airfoil[-1,1] = mean
    return airfoil

def optimize_latent(syn_func, dim, bounds, n_eval, run_id):
    # Optimize in the latent space
    n_pre_samples = 10
    kernel = gp.kernels.Matern()
    gp_model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=100, normalize_y=False)
    zp = []
    perfs = []
    opt_perfs = []
    s = 1.0
    gamma = 0.99
    for i in range(n_eval):
        if len(zp) < n_pre_samples:
            z = np.random.uniform(bounds[:,0], bounds[:,1], size=dim)
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
#            z = sample_next_point(dim, neg_expected_improvement, gp_model, previous_optimum, bounds, n_restarts=100)
            z = sample_next_point(dim, neg_expected_improvement, gp_model, previous_optimum, bounds=None, 
                                  random_search=100000, previous_point=previous_point, scale=s)
            s *= gamma
        
        x = syn_func(z)
        perf = evaluate(x)
        z = np.squeeze(z)
        zp.append(z)
        perfs.append(perf)
        opt_idx = np.argmax(perfs)
        opt_z = zp[opt_idx]
        opt_perf = perfs[opt_idx]
        opt_perfs.append(opt_perf) # Best performance so far
        print('GAN-2-GA {}-{}: CL/CD {:.2f} best-so-far {:.2f}'.format(run_id, i+1, perf, opt_perf))
        
    opt_z = opt_z.reshape(1,-1)
    opt_airfoil = syn_func(opt_z)
    print('Optimal CL/CD {}'.format(opt_perfs[-1]))
        
    return opt_airfoil, opt_perfs, opt_z

def optimize_overall(x0, syn_func, perturb_type, perturb, n_eval, run_id):
    # Optimize in the nosie space
    n_best = 30
    n_random = 10
    n_children = 5
    chance_of_mutation = 0.1
    population_size = int((n_best+n_random)/2*n_children)
    population = generate_first_population(x0, population_size, perturb_type, perturb)
    best_inds = []
    best_perfs = []
    opt_perfs = []
    i = 0
    while 1:
        breeders, best_perf, best_individual = select(population, n_best, n_random, syn_func)
        best_inds.append(best_individual)
        best_perfs.append(best_perf)
        opt_perfs += [np.max(best_perfs)] * population_size # Best performance so far
        print('GAN-2-GA %d-%d: fittest %.2f' % (run_id, i+1, best_perf))
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
    parser.add_argument('--n_eval', type=int, default=1000, help='number of total evaluations per run')
    parser.add_argument('--n_eval_latent', type=int, default=100, help='number of evaluations in the latent space per run')
    args = parser.parse_args()
    
    n_runs = args.n_runs
    n_eval = args.n_eval
    n_eval_latent = args.n_eval_latent
    n_eval_overall = n_eval - n_eval_latent
    
    # Airfoil parameters
    latent_dim = 3
    noise_dim = 10
    n_points = 192
    bezier_degree = 32
    latent_bounds = (0., 1.)
    latent_bounds = np.tile(latent_bounds, (latent_dim,1))
    
    # Restore trained model
    model = GAN(latent_dim, noise_dim, n_points, bezier_degree, latent_bounds)
    model.restore()
    
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    for i in range(n_runs):
        start_time = time.time()
        # Optimize in the latent space
        syn_func = lambda x: synthesize(x.reshape(1,-1), model)
        opt_airfoil, opt_perfs_latent, opt_latent = optimize_latent(syn_func, latent_dim, latent_bounds, n_eval_latent, i+1)
        # Optimize in both latent and noise space
        noise0 = np.zeros(noise_dim)
        syn_func = lambda x: synthesize(x[:latent_dim].reshape(1,-1), model, x[latent_dim:].reshape(1,-1))
        perturb_type = 'absolute'
        perturb = np.append(0.1*np.ones(latent_dim), 1.0*np.ones(noise_dim))
        x0 = np.append(np.squeeze(opt_latent), noise0)
        opt_airfoil, opt_perfs_overall = optimize_overall(x0, syn_func, perturb_type, perturb, n_eval_overall, i+1)
        end_time = time.time()
        opt_airfoil_runs.append(opt_airfoil)
        opt_perfs_runs.append([0] + opt_perfs_latent + opt_perfs_overall)
        time_runs.append(end_time-start_time)
    
    opt_airfoil_runs = np.array(opt_airfoil_runs)
    opt_perfs_runs = np.array(opt_perfs_runs)
    np.save('opt_results/gan_2_ga/opt_airfoil.npy', opt_airfoil_runs)
    np.save('opt_results/gan_2_ga/opt_history.npy', opt_perfs_runs)
    
    # Plot optimization history
    mean_perfs_runs = np.mean(opt_perfs_runs, axis=0)
    plt.figure()
    plt.plot(np.arange(n_eval+1, dtype=int), mean_perfs_runs)
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Optimal CL/CD')
#    plt.xticks(np.linspace(0, n_eval+1, 5, dtype=int))
    plt.savefig('opt_results/gan_2_ga/opt_history.svg')
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
    plt.savefig('opt_results/gan_2_ga/opt_airfoil.svg')
    plt.close()

    print 'GAN-2-GA completed :)'
