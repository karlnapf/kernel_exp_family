from kernel_exp_family.kernels.kernels import gaussian_kernel
import numpy as np


def compute_b_sym(Z, K, sigma):
    assert K.shape[0] == Z.shape[0]
    assert K.shape[0] == K.shape[1]
    
    
    D = Z.shape[1]
    N = Z.shape[0]
    
    b = np.zeros(N)
    K1 = np.sum(K, 1)
    for l in np.arange(D):
        x_l = Z[:, l]
        s_l = x_l ** 2
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_s_K = s_l[:, np.newaxis] * K
        D_x_K = x_l[:, np.newaxis] * K
        
        b += 2. / sigma * (K.dot(s_l) \
                        + np.sum(D_s_K, 1) \
                        - 2 * D_x_K.dot(x_l)) - K1
    return b

def compute_C_sym(Z, K, sigma):
    assert K.shape[0] == Z.shape[0]
    assert K.shape[0] == K.shape[1]
    
    D = Z.shape[1]
    N = Z.shape[0]
    
    C = np.zeros((N, N))
    for l in np.arange(D):
        x_l = Z[:, l]
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_x_K = x_l[:, np.newaxis] * K
        K_D_x = K * x_l
        
        C += (D_x_K - K_D_x).dot(K_D_x - D_x_K)
    
    return C

def fit_sym(Z, sigma, lmbda, K=None, b=None, C=None):
        # compute quantities
        if K is None:
            K = gaussian_kernel(Z, sigma=sigma)
        
        if b is None:
            b = compute_b_sym(Z, K, sigma)
        
        if C is None:
            C = compute_C_sym(Z, K, sigma)

        # solve regularised linear system
        a = -sigma / 2. * np.linalg.solve(C + (K + np.eye(len(C))) * lmbda,
                                          b)
        
        return a

def objective_sym(Z, sigma, lmbda, alpha, K=None, b=None, C=None):
    if K is None and ((b is None or C is None) or lmbda > 0):
        K = gaussian_kernel(Z, sigma=sigma)
    
    if C is None:
        C = compute_C_sym(Z, K, sigma)
    
    if b is None:
        b = compute_b_sym(Z, K, sigma)
    
    N = len(Z)
    first = 2. / (N * sigma) * alpha.dot(b)
    second = 2. / (N * sigma ** 2) * alpha.dot(
                                               (C + (K + np.eye(len(C))) * lmbda).dot(alpha)
                                               )
    J = first + second
    return J

