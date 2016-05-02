import time

import kernel_exp_family.estimators.full.gaussian_nystrom as python_nystrom
import numpy as np


import pyximport; pyximport.install()
try:
    import kernel_exp_family.estimators.full.gaussian_nystrom_cython as cython_nystrom
except ImportError:
    print("Skipping as needs Cython")
    exit()



if __name__ == "__main__":
    sigma = 3.
    lmbda = 0.01
    N = 100
    D = 2
    fraction_subsampling = 0.3
    m = np.int(np.round((N * D) * fraction_subsampling))
    
    X = np.random.randn(N, D)
    inds = np.random.permutation(N * D)[:m]
    
    start = time.time()
    python_nystrom.build_system_nystrom(X, sigma, lmbda, inds)
    end = time.time()
    print "python:", end - start
    
    start = time.time()
    cython_nystrom.build_system_nystrom(X, sigma, lmbda, inds)
    end = time.time()
    print "cython:", end - start
    
    
