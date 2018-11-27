
"""
Trains an Hmodel, and visulizes results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module

from gan import GAN
from shape_plot import plot_samples, plot_grid
from manifold_evaluation.diversity import ci_rdiv
from manifold_evaluation.likelihood import ci_mll
from manifold_evaluation.consistency import ci_cons
from manifold_evaluation.smoothness import ci_rsmth
from utils import ElapsedTimer


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='startover', help='startover, continue, or evaluate')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['startover', 'continue', 'evaluate']
    
    latent_dim = 3
    noise_dim = 10
    bezier_degree = 31
    train_steps = 20000
    batch_size = 32
    symm_axis = None
    bounds = (0., 1.)
    
    # Read dataset
    data_fname = 'airfoil_interp.npy'
    X = np.load(data_fname)
    
    print('Plotting training samples ...')
    samples = X[np.random.choice(range(X.shape[0]), size=36)]
#    plot_samples(None, samples, scatter=True, symm_axis=symm_axis, s=1.5, alpha=.7, c='k', fname='samples')
    plot_samples(None, samples, scale=1.0, scatter=False, symm_axis=symm_axis, lw=1.2, alpha=.7, c='k', fname='samples')
    
    # Split training and test data
    test_split = 0.8
    N = X.shape[0]
    split = int(N*test_split)
    X_train = X[:split]
    X_test = X[split:]
    
    # Train
    model = GAN(latent_dim, noise_dim, X_train.shape[1], bezier_degree, bounds)
    if args.mode == 'startover':
        timer = ElapsedTimer()
        model.train(X_train, batch_size=batch_size, train_steps=train_steps, save_interval=args.save_interval)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
        runtime_file = open('gan/runtime.txt', 'w')
        runtime_file.write('%s\n' % runtime_mesg)
        runtime_file.close()
    else:
        model.restore()
    
    print('Plotting synthesized shapes ...')
    plot_grid(5, gen_func=model.synthesize, d=latent_dim, bounds=bounds, scale=1.0, scatter=False, symm_axis=symm_axis, 
              alpha=.7, lw=1.2, c='k', fname='gan/synthesized')
    
    n_runs = 10
    
    mll_mean, mll_err = ci_mll(n_runs, model.synthesize, X_test)
    rdiv_mean, rdiv_err = ci_rdiv(n_runs, X, model.synthesize)
    cons_mean, cons_err = ci_cons(n_runs, model.synthesize, latent_dim, bounds) # Only for GANs
    rsmth_mean, rsmth_err = ci_rsmth(n_runs, model.synthesize, X_test)
    
    results_mesg_1 = 'Mean log likelihood: %.1f +/- %.1f' % (mll_mean, mll_err)
    results_mesg_2 = 'Relative diversity: %.3f +/- %.3f' % (rdiv_mean, rdiv_err)
    results_mesg_3 = 'Consistency: %.3f +/- %.3f' % (cons_mean, cons_err)
    results_mesg_4 = 'Smoothness: %.3f +/- %.3f' % (rsmth_mean, rsmth_err)
    
    results_file = open('gan/results.txt', 'w')
    
    print(results_mesg_1)
    results_file.write('%s\n' % results_mesg_1)
    print(results_mesg_2)
    results_file.write('%s\n' % results_mesg_2)
    print(results_mesg_3)
    results_file.write('%s\n' % results_mesg_3)
    print(results_mesg_4)
    results_file.write('%s\n' % results_mesg_4)
    
#    rdiv_means = []
#    rdiv_errs = []
#    for k in range(latent_dim):
#        rdiv_mean_k, rdiv_err_k = ci_rdiv(100, X, model.synthesize, latent_dim, k, plot_bounds)
#        rdiv_means.append(rdiv_mean_k)
#        rdiv_errs.append(rdiv_err_k)
#        results_mesg_k = 'Relative diversity for latent dimension %d: %.3f +/- %.3f' % (k, rdiv_mean_k, rdiv_err_k)
#        print(results_mesg_k)
#        results_file.write('%s\n' % results_mesg_k)
        
    results_file.close()
    
#    plt.figure()
#    plt.errorbar(np.arange(latent_dim)+1, rdiv_means, yerr=rdiv_err_k)
#    plt.xlabel('Latent Dimensions')
#    plt.ylabel('Relative diversity')
#    plt.savefig('rdiv.svg', dpi=600)

    print 'All completed :)'
