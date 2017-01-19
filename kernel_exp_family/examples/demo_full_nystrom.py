from kernel_exp_family.estimators.full.gaussian import KernelExpFullGaussian
from kernel_exp_family.examples.tools import visualise_fit_2d
import matplotlib.pyplot as plt
import numpy as np

def get_full_instance(D, N):
    # arbitrary choice of parameters here
    sigma = 5.
    lmbda = 0.01
    return KernelExpFullGaussian(sigma, lmbda, D)

def get_nystrom_instance(D, N, X):
    # arbitrary choice of parameters here
    sigma = 15.
    lmbda = 0.01
    m=50
    basis = X[np.random.permutation(len(X))[:m]]
    return KernelExpFullGaussian(sigma, lmbda, D, basis=basis)

if __name__ == '__main__':
    N = 200
    D = 2
    
    # fit model to samples from a standard Gaussian
    X = np.random.randn(N, D)
    bananicity=0.03
    V=100.0
    X[:, 0] = np.sqrt(V) * X[:, 0]
    X[:, 1] = X[:, 1] + bananicity * (X[:, 0] ** 2 - V)
    
    # estimator API object, try different estimators here
    est = get_nystrom_instance(D, N, X)
    
    est.fit(X)
    
    print est.objective(X)
    
    Xs = np.linspace(-20, 20, 20)
    Ys = np.linspace(-7, 12, 20)
    visualise_fit_2d(est, X, Xs, Ys)
    
    plt.show()
