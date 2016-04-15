from kernel_exp_family.estimators.full.gaussian import KernelExpFullGaussian
from kernel_exp_family.estimators.full.gaussian_nystrom import KernelExpFullNystromGaussian
from kernel_exp_family.examples.tools import visualise_fit_2d
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    sigma = 3
    lmbda = 0.01
    N = 100
    D = 2
    m = 10
    
    X = np.random.randn(N,D)
    x = np.zeros(D)
    
    est = KernelExpFullNystromGaussian(sigma, lmbda, D, N, m=N*D*0.3)
    est.fit(X)
    visualise_fit_2d(est, X)
    plt.suptitle("m={}".format(est.m))
    
    est_full = KernelExpFullGaussian(sigma, lmbda, D, N)
    est_full.fit(X)
    visualise_fit_2d(est_full, X)
    
    plt.show()
