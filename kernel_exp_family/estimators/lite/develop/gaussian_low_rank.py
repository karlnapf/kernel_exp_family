from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve.iterative import bicgstab

import numpy as np


def compute_b_sym(Z, L, sigma):
    assert len(Z) == len(L)
    
    D = Z.shape[1]
    N = Z.shape[0]
    
    b = np.zeros(N)
    L1 = L.dot(np.sum(L.T, 1))
    for l in np.arange(D):
        x_l = Z[:, l]
        s_l = x_l ** 2
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_s_L = s_l[:, np.newaxis] * L
        D_x_L = x_l[:, np.newaxis] * L
        
        # compute b incrementally exploiting the Cholesky factorisation of K
        b += 2. / sigma * (L.dot(L.T.dot(s_l)) \
                        + D_s_L.dot(np.sum(L.T, 1)) \
                        - 2 * D_x_L.dot(L.T.dot(x_l))) - L1
    
    return b

def apply_left_C_sym(v, Z, L, lmbda):
    assert len(Z) == len(L)
    assert len(v.shape) == 1
    
    N = Z.shape[0]
    D = Z.shape[1]
    
    # multiply C to v (C is a sum over d=1...D)
    result = np.zeros(N)
    for l in range(D):
        x_l = Z[:, l]
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_x_L = x_l[:, np.newaxis] * L
        L_T_D_x = L.T * x_l
        
        # right term of product
        x = L.T.dot(v)
        x = D_x_L.dot(x)
        y = L_T_D_x.dot(v)
        y = L.dot(y)
        
        # right term times v
        temp = x - y
        
        # term of product
        x = L_T_D_x.dot(temp)
        x = L.dot(x)
        y = L.T.dot(temp)
        y = D_x_L.dot(y)
        
        # add both terms times v to result
        result += x - y
    
    if lmbda > 0:
        # regularise with K=L_X.dot(L_X.T)
        result += lmbda * L.dot(L.T.dot(v))
        
        # regularise with I
        result += lmbda * v
    
    return result

def fit_sym(Z, sigma, lmbda, L,
                                                    cg_tol=1e-3,
                                                    cg_maxiter=None):
        if cg_maxiter is None:
            # CG needs at max dimension many iterations
            cg_maxiter = L.shape[0]
        
        N = Z.shape[0]
        
        # set up and solve regularised linear system via bicgstab
        # this never stores an NxN matrix
        b = compute_b_sym(Z, L, sigma)
        matvec = lambda v:apply_left_C_sym(v, Z, L, lmbda)
        C_operator = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
        solution, info = bicgstab(C_operator, b, tol=cg_tol, maxiter=cg_maxiter)
        if info > 0:
            print "Warning: CG not terminated within specified %d iterations" % cg_maxiter
        a = -sigma / 2. * solution
        
        return a


def objective_sym(Z, sigma, lmbda, alpha, L, b=None):
    if b is None:
        b = compute_b_sym(Z, L, sigma)
    
    N = len(Z)
    first = 2. / (N * sigma) * alpha.dot(b)
    second = 2. / (N * sigma ** 2) * alpha.dot(
                                               apply_left_C_sym(
                                                                          alpha, Z, L, lmbda
                                                                          )
                                               )
    J = first + second
    return J

