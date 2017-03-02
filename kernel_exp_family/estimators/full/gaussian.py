from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.tools.assertions import assert_array_shape
from kernel_exp_family.kernels.kernels import gaussian_kernel_hessians, \
    gaussian_kernel_dx_dx_dy, gaussian_kernel_dx_dx_dy_dy, gaussian_kernel_grad, \
    gaussian_kernel_dx_dx, gaussian_kernel_dx_i_dx_i_dx_j, gaussian_kernel_dx_i_dx_j, \
    gaussian_kernel_dx_i_dx_i_dx_j_dx_j

import numpy as np
import scipy as sp

def compute_h(basis, data, sigma):
    n, d = data.shape
    m, _ = basis.shape
    
    h = np.zeros((n, d))
    for _, x_a in enumerate(basis):
        for b, x_b in enumerate(data):
            h[b, :] += np.sum(gaussian_kernel_dx_dx_dy(x_a, x_b, sigma), axis=0)
    
    # note: the missing division by N_data is done further downstream
    return h.reshape(-1) / m

def compute_xi_norm_2(basis, data, sigma):
    n, _ = data.shape
    m, _ = basis.shape
    norm_2 = 0.
    for _, x_a in enumerate(basis):
        for _, x_b in enumerate(data):
            norm_2 += np.sum(gaussian_kernel_dx_dx_dy_dy(x_a, x_b, sigma))
    
    return norm_2 / (n * m)

def fit(basis, X, sigma, lmbda):
    m, d = basis.shape
    n, _ = X.shape
    
    G = gaussian_kernel_hessians(X=basis, Y=X, sigma=sigma)
    h = compute_h(basis, X, sigma)
    if basis is X:
        # full estimator, no approximation
        np.fill_diagonal(G, np.diag(G) + n * lmbda)
        
        cho_lower = sp.linalg.cho_factor(G)
        beta = sp.linalg.cho_solve(cho_lower, h / lmbda)
    else:
        # e.g. nystrom estimator
        G_mn = G
        
        # TODO if sub-sampling is used, can avoid the re-computing here
        G_mm = gaussian_kernel_hessians(X=basis, Y=basis, sigma=sigma)
        
        # TODO compare least squares pinv vs SVD pinv2
        G_dagger = sp.linalg.pinv2(np.dot(G_mn, G_mn.T) + lmbda * n * G_mm)
        
        beta = np.dot(G_dagger, np.dot(G_mn, h / lmbda))
    
    return beta.reshape(m, d)

def log_pdf(x, basis, sigma, lmbda, beta):
    m, D = basis.shape
    assert_array_shape(x, ndim=1, dims={0: D})
    
    xi = 0
    betasum = 0
    for a in range(m):
        x_a = np.atleast_2d(basis[a])
        xi += np.sum(gaussian_kernel_dx_dx(x, x_a, sigma)) / m
        gradient_x_xa = gaussian_kernel_grad(x, x_a, sigma)
        betasum += np.dot(gradient_x_xa, beta[a, :])
    
    return np.float(-1.0 / lmbda * xi + betasum)

def grad(x, basis, sigma, lmbda, beta):
    m, D = basis.shape
    assert_array_shape(x, ndim=1, dims={0: D})
    
    xi_grad = 0
    betasum_grad = 0
    for a, x_a in enumerate(basis):
        xi_grad += np.sum(gaussian_kernel_dx_i_dx_i_dx_j(x, x_a, sigma), axis=0) / m
        left_arg_hessian = gaussian_kernel_dx_i_dx_j(x, x_a, sigma)
        betasum_grad += beta[a, :].dot(left_arg_hessian)

    return -1.0 / lmbda * xi_grad + betasum_grad

def second_order_grad(x, basis, sigma, lmbda, beta):
    """ Computes $\frac{\partial^2 log p(x)}{\partial x_i^2} """
    m, D = basis.shape
    assert_array_shape(x, ndim=1, dims={0: D})

    xi_grad = 0
    betasum_grad = 0
    for a, x_a in enumerate(basis):
        xi_grad += np.sum(gaussian_kernel_dx_i_dx_i_dx_j_dx_j(x, x_a, sigma),
                          axis=0) / m
        left_arg_hessian = gaussian_kernel_dx_i_dx_i_dx_j(x, x_a, sigma)
        betasum_grad += beta[a, :].dot(left_arg_hessian)

    return -1.0 / lmbda * xi_grad + betasum_grad

def compute_objective(X, basis, sigma, lmbda, beta):
    N_test, _ = X.shape

    objective = 0.0

    for _, x_a in enumerate(X):
        g = grad(x_a, basis, sigma, lmbda, beta)
        g2 = second_order_grad(x_a, basis, sigma, lmbda, beta)
        objective += (0.5 * np.dot(g, g) + np.sum(g2)) / N_test

    return objective

class KernelExpFullGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, D, basis=None):
        self.sigma = sigma
        self.lmbda = lmbda
        self.D = D
        self.basis = basis
        
        # initial RKHS function is flat
        self.beta = 0
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        if self.basis is None:
            self.basis = X
            
        self.beta = fit(self.basis, X, self.sigma, self.lmbda)
    
    def log_pdf(self, x):
        return log_pdf(x, self.basis, self.sigma, self.lmbda, self.beta)

    def grad(self, x):
        return grad(x, self.basis, self.sigma, self.lmbda, self.beta)

    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        return compute_objective(X, self.basis, self.sigma, self.lmbda, self.beta)

    def get_parameter_names(self):
        return ['sigma', 'lmbda']
