
"""
Compare preformance of methods within certain running time.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Optimize')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--n_eval', type=int, default=1000, help='number of evaluations per run')
    args = parser.parse_args()
    
    n_runs = args.n_runs
    n_eval = args.n_eval
    
    ''' Call optimization '''
    os.system('python optimize_gan_bo.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    os.system('python optimize_pca_bo.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    os.system('python optimize_nurbs_bo.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    os.system('python optimize_parsec_bo.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    os.system('python optimize_nurbs_ga.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    os.system('python optimize_parsec_ga.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    
    ''' Plot history '''
    gan_bo_hist = np.load('opt_results/gan_bo/opt_history.npy')
    pca_bo_hist = np.load('opt_results/pca_bo/opt_history.npy')
    nurbs_bo_hist = np.load('opt_results/nurbs_bo/opt_history.npy')
    nurbs_ga_hist = np.load('opt_results/nurbs_ga/opt_history.npy')
    parsec_bo_hist = np.load('opt_results/parsec_bo/opt_history.npy')
    parsec_ga_hist = np.load('opt_results/parsec_ga/opt_history.npy')
    
    mean_gan_bo_hist = np.mean(gan_bo_hist, axis=0)
    var_gan_bo_hist = np.var(gan_bo_hist, axis=0)
    mean_pca_bo_hist = np.mean(pca_bo_hist, axis=0)
    var_pca_bo_hist = np.var(pca_bo_hist, axis=0)
    mean_nurbs_bo_hist = np.mean(nurbs_bo_hist, axis=0)
    var_nurbs_bo_hist = np.var(nurbs_bo_hist, axis=0)
    mean_nurbs_ga_hist = np.mean(nurbs_ga_hist, axis=0)
    var_nurbs_ga_hist = np.var(nurbs_ga_hist, axis=0)
    mean_parsec_bo_hist = np.mean(parsec_bo_hist, axis=0)
    var_parsec_bo_hist = np.var(parsec_bo_hist, axis=0)
    mean_parsec_ga_hist = np.mean(parsec_ga_hist, axis=0)
    var_parsec_ga_hist = np.var(parsec_ga_hist, axis=0)
    
    linestyles = ['-', '--', ':', '-.', (0, (5,1,1,1,1,1)), (0, (1,4))]
    lss = itertools.cycle(linestyles)
    
    plt.figure(figsize=(7,5))
    plt.plot(np.arange(n_eval+1, dtype=int), mean_gan_bo_hist, ls=next(lss), label=r'B$\acute{e}$zierGAN+EGO')
    plt.plot(np.arange(n_eval+1, dtype=int), mean_pca_bo_hist, ls=next(lss), label='PCA+EGO')
    plt.plot(np.arange(n_eval+1, dtype=int), mean_nurbs_bo_hist, ls=next(lss), label='NURBS+EGO')
    plt.plot(np.arange(n_eval+1, dtype=int), mean_nurbs_ga_hist, ls=next(lss), label='NURBS+GA')
    plt.plot(np.arange(n_eval+1, dtype=int), mean_parsec_bo_hist, ls=next(lss), label='PARSEC+EGO')
    plt.plot(np.arange(n_eval+1, dtype=int), mean_parsec_ga_hist, ls=next(lss), label='PARSEC+GA')
    plt.legend()
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Optimal CL/CD')
    plt.ylim(ymax=400)
#    plt.xticks(np.linspace(0, max_n_eval+1, 5, dtype=int))
    plt.savefig('opt_results/opt_history.svg')
    plt.close()
    
    ''' Plot optimal solutions '''
    gan_bo_opt = np.load('opt_results/gan_bo/opt_airfoil.npy')
    pca_bo_opt = np.load('opt_results/pca_bo/opt_airfoil.npy')
    nurbs_bo_opt = np.load('opt_results/nurbs_bo/opt_airfoil.npy')
    nurbs_ga_opt = np.load('opt_results/nurbs_ga/opt_airfoil.npy')
    parsec_bo_opt = np.load('opt_results/parsec_bo/opt_airfoil.npy')
    parsec_ga_opt = np.load('opt_results/parsec_ga/opt_airfoil.npy')
    
    # Separate plots
    def subplot_airfoil(position, airfoils, title):
        plt.subplot(position)
        for airfoil in airfoils:
            plt.plot(airfoil[:,0], airfoil[:,1], '-', c='k', alpha=1.0/n_runs)
        plt.title(title)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.2, 0.2])
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
    
    plt.figure(figsize=(16, 5))
    subplot_airfoil(231, gan_bo_opt, r'B$\acute{e}$zierGAN+EGO')
    subplot_airfoil(232, nurbs_bo_opt, 'NURBS+EGO')
    subplot_airfoil(233, nurbs_ga_opt, 'NURBS+GA')
    subplot_airfoil(234, pca_bo_opt, 'PCA+EGO')
    subplot_airfoil(235, parsec_bo_opt, 'PARSEC+EGO')
    subplot_airfoil(236, parsec_ga_opt, 'PARSEC+GA')
    plt.tight_layout()
    plt.savefig('opt_results/opt_airfoils.svg')
    plt.close()