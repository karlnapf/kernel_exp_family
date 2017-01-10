from abc import abstractmethod

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.tools.assertions import assert_array_shape
from kernel_exp_family.kernels.kernels import gaussian_kernel_hessians, \
    gaussian_kernel_dx_dx_dy, gaussian_kernel_dx_dx_dy_dy, gaussian_kernel_grad, \
    gaussian_kernel_dx_dx, gaussian_kernel_dx_i_dx_i_dx_j, gaussian_kernel_dx_i_dx_j, \
    gaussian_kernel_dx_i_dx_i_dx_j_dx_j

import numpy as np


def compute_h(data, sigma):
    n, d = data.shape
    h = np.zeros((n, d))
    for b, x_b in enumerate(data):
        for _, x_a in enumerate(data):
            h[b, :] += np.sum(gaussian_kernel_dx_dx_dy(x_a, x_b, sigma), axis=0)
            
    return h / n


def compute_lower_right_submatrix(all_hessians, N, lmbda):
    return np.dot(all_hessians, all_hessians) / N + lmbda * all_hessians

def compute_first_row(h, all_hessians, n, lmbda):
    return np.dot(h, all_hessians) / n + lmbda * h

def compute_RHS(h, xi_norm_2):
    b = np.zeros(h.size + 1)
    b[0] = -xi_norm_2
    b[1:] = -h.reshape(-1)

    return b

def compute_xi_norm_2(data, sigma):
    n, _ = data.shape
    norm_2 = 0
    for _, x_a in enumerate(data):
        for _, x_b in enumerate(data):
            norm_2 += np.sum(gaussian_kernel_dx_dx_dy_dy(x_a, x_b, sigma))
            
    return norm_2 / n ** 2


def build_system(X, sigma, lmbda):
    n, d = X.shape
    
    h = compute_h(X, sigma).reshape(-1)
    all_hessians = gaussian_kernel_hessians(X, sigma=sigma)
    xi_norm_2 = compute_xi_norm_2(X, sigma)
    
    A = np.zeros((n * d + 1, n * d + 1))
    A[0, 0] = np.dot(h, h) / n + lmbda * xi_norm_2
    A[1:, 1:] = compute_lower_right_submatrix(all_hessians, n, lmbda)
    
    A[0, 1:] = compute_first_row(h, all_hessians, n, lmbda)
    A[1:, 0] = A[0, 1:]
    
    b = compute_RHS(h, xi_norm_2)
    
    return A, b

def fit(X, sigma, lmbda):
    n, d = X.shape
    A, b = build_system(X, sigma, lmbda)
    x = np.linalg.solve(A, b)
    alpha = x[0]
    beta = x[1:].reshape(n, d)
    return alpha, beta

def log_pdf(x, X, sigma, alpha, beta):
    _, D = X.shape
    assert_array_shape(x, ndim=1, dims={0: D})
    N = len(X)
    
    SE_dx_dx_l = lambda x, y : gaussian_kernel_dx_dx(x, y.reshape(1, -1), sigma)
    SE_dx_l = lambda x, y: gaussian_kernel_grad(x, y.reshape(1, -1), sigma)
    
    xi = 0
    betasum = 0
    for a in range(N):
        x_a = X[a, :]
        xi += np.sum(SE_dx_dx_l(x, x_a)) / N
        gradient_x_xa = np.squeeze(SE_dx_l(x, x_a))
        betasum += np.dot(gradient_x_xa, beta[a, :])
    
    return np.float(alpha * xi + betasum)

def grad(x, X, sigma, alpha, beta):
    N, D = X.shape
    assert_array_shape(x, ndim=1, dims={0: D})
    
    xi_grad = 0
    betasum_grad = 0
    for a, x_a in enumerate(X):
        xi_grad += np.sum(gaussian_kernel_dx_i_dx_i_dx_j(x, x_a, sigma), axis=0) / N
        left_arg_hessian = gaussian_kernel_dx_i_dx_j(x, x_a, sigma)
        betasum_grad += beta[a, :].dot(left_arg_hessian)

    return alpha * xi_grad + betasum_grad


def second_order_grad(x, X, sigma, alpha, beta):
    """ Computes $\frac{\partial^2 log p(x)}{\partial x_i^2} """
    N, D = X.shape
    assert_array_shape(x, ndim=1, dims={0: D})

    xi_grad = 0
    betasum_grad = 0
    for a, x_a in enumerate(X):
        xi_grad += np.sum(gaussian_kernel_dx_i_dx_i_dx_j_dx_j(x, x_a, sigma),
                          axis=0) / N
        left_arg_hessian = gaussian_kernel_dx_i_dx_i_dx_j(x, x_a, sigma)
        betasum_grad += beta[a, :].dot(left_arg_hessian)

    return alpha * xi_grad + betasum_grad


def compute_objective(X_test, X_train, sigma, alpha, beta):
    N_test, D = X_test.shape

    objective = 0.0

    for a, x_a in enumerate(X_test):
        g = grad(x_a, X_train, sigma, alpha, beta)
        g2 = second_order_grad(x_a, X_train, sigma, alpha, beta)
        objective += (0.5 * np.dot(g, g) + np.sum(g2)) / N_test

    return objective


class KernelExpFullGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, D, N):
        self.sigma = sigma
        self.lmbda = lmbda
        self.D = D
        self.N = N
        
        # initial RKHS function is flat
        self.alpha = 0
        self.beta = np.zeros(D * N)
        self.X = np.zeros((0, D))
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        self.fit_wrapper()
    
    @abstractmethod
    def fit_wrapper_(self):
        self.alpha, self.beta = fit(self.X, self.sigma, self.lmbda)
    
    def log_pdf(self, x):
        return log_pdf(x, self.X, self.sigma, self.alpha, self.beta)

    def grad(self, x):
        return grad(x, self.X, self.sigma, self.alpha, self.beta)

    def log_pdf_multiple(self, X):
        return np.array([self.log_pdf(x) for x in X])
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        return compute_objective(X, self.X, self.sigma, self.alpha, self.beta)

    def get_parameter_names(self):
        return ['sigma', 'lmbda']
