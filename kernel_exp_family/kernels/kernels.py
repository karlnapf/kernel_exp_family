from scipy.spatial.distance import squareform, pdist, cdist

import numpy as np


def gaussian_kernel(X, Y=None, sigma=1.):
    assert(len(X.shape) == 2)
    
    # if X==Y, use more efficient pdist call which exploits symmetry
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert(len(Y.shape) == 2)
        assert(X.shape[1] == Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')
        
    K = np.exp(-(sq_dists) / sigma)
    return K

def gaussian_kernel_grad(x, Y, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(Y.shape) == 2)
    assert(len(x) == Y.shape[1])
    
    x_2d = x[np.newaxis, :]
    k = gaussian_kernel(x_2d, Y, sigma)
    differences = Y - x
    G = (2.0 / sigma) * (k.T * differences)
    return G
