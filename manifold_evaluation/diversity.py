import numpy as np
from utils import mean_err


def variance(X):
    cov = np.cov(X.T)
    var = np.trace(cov)/cov.shape[0]
#    var = np.mean(np.var(X, axis=0))
#    var = np.linalg.det(cov)
#    var = var**(1./cov.shape[0])
    return var

def rdiv(X_train, X_gen):
    ''' Relative div '''
    X_train = np.squeeze(X_train)
#    train_div = np.sum(np.var(X_train, axis=0))
#    gen_div = np.sum(np.var(X_gen, axis=0))
    X_train = X_train.reshape((X_train.shape[0], -1))
    train_div = variance(X_train)
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    gen_div = variance(X_gen)
#    n = 100
#    gen_div = train_div = 0
#    for i in range(n):
#        a, b = np.random.choice(X_gen.shape[0], 2, replace=False)
#        gen_div += np.linalg.norm(X_gen[a] - X_gen[b])
#        c, d = np.random.choice(X_train.shape[0], 2, replace=False)
#        train_div += np.linalg.norm(X_train[c] - X_train[d])
    rdiv = gen_div/train_div
    return rdiv

def ci_rdiv(n, X_train, gen_func, d=None, k=None, bounds=None):
    rdivs = np.zeros(n)
    for i in range(n):
        if d is None or k is None or bounds is None:
            X_gen = gen_func(X_train.shape[0])
        else:
            latent = np.random.uniform(bounds[0], bounds[1])*np.ones((X_train.shape[0], d))
            latent[:, k] = np.random.uniform(bounds[0], bounds[1], size=X_train.shape[0])
            X_gen = gen_func(latent)
#            from shape_plot import plot_samples
#            plot_samples(None, X_gen[:10], scatter=True, s=1, alpha=.7, c='k', fname='gen_%d' % k)
        rdivs[i] = rdiv(X_train, X_gen)
    mean, err = mean_err(rdivs)
    return mean, err
