from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.estimators.full.develop.gaussian_nystrom import nystrom_naive,\
    ind_to_ai
from kernel_exp_family.kernels.kernels import gaussian_kernel_dx_component,\
    gaussian_kernel_dx_dx_component, gaussian_kernel_dx_i_dx_i_dx_j_component,\
    gaussian_kernel_dx_i_dx_j_component
from kernel_exp_family.tools.assertions import assert_array_shape
import numpy as np


def nystrom(X, sigma, lmbda, inds):
    A, b = build_system(X, sigma, lmbda)
    
    inds_with_xi = np.zeros(len(inds)+1)
    inds_with_xi[1:] = inds+1
    
    A_mm = A[:, inds_with_xi][inds_with_xi]
    A_nm = A[:, inds_with_xi]
    b_m = b[inds_with_xi]
    
    return A_mm, A_nm, b_m, inds

def fit(X, sigma, lmbda, inds):
    A_nm, b = nystrom(X, sigma, lmbda, inds)
    
    A = np.dot(A_nm.T, A_nm)
    b = np.dot(A_nm.T, b).flatten()
    
    x = np.linalg.solve(A, b)
    alpha = x[0]
    beta = x[1:]
    return alpha, beta

def log_pdf(x, X, sigma, alpha, beta, inds):
    N, D = X.shape
    
    xi = 0
    betasum = 0
    
    ais = [ind_to_ai(ind, D) for ind in range(len(inds))]
    
    for ind, (a, i) in enumerate(ais):
        gradient_x_xa_i = gaussian_kernel_dx_component(x, X[a, :], i, sigma)
        xi_grad_i = gaussian_kernel_dx_dx_component(x, X[a,:], i, sigma)
        
        xi += xi_grad_i / N
        betasum += gradient_x_xa_i * beta[ind]
    
    return np.float(alpha * xi + betasum)

def grad(x, X, sigma, alpha, beta, inds):
    N, D = X.shape
    
    xi_grad = 0
    betasum_grad = 0
    
    ais = [ind_to_ai(ind, D) for ind in range(len(inds))]
    
    for ind, (a,i) in enumerate(ais):
        x_a = X[a]
        xi_gradient_mat_component = gaussian_kernel_dx_i_dx_i_dx_j_component(x, x_a, i, sigma)
        left_arg_hessian_component = gaussian_kernel_dx_i_dx_j_component(x, x_a, i, sigma)
        
        xi_grad += xi_gradient_mat_component / N
        betasum_grad += beta[ind] * left_arg_hessian_component

    return alpha * xi_grad + betasum_grad

class KernelExpFullNystromGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, D, N, m):
        self.sigma = sigma
        self.lmbda = lmbda
        self.N = N
        self.D = D
        
        # initial RKHS function is flat
        self.alpha = 0
        self.beta = np.zeros(m)
        self.X = np.zeros((0, D))
        
        self.inds = np.sort(np.random.permutation(N * D)[:m])
        self.m = m
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={0: self.N, 1: self.D})
        self.X = X
        self.alpha, self.beta = fit(self.X, self.sigma, self.lmbda, self.inds)
    
    def log_pdf(self, x):
        return log_pdf(x, self.X, self.sigma, self.alpha, self.beta, self.inds)

    def grad(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
        return grad(x, self.X, self.sigma, self.alpha, self.beta, self.inds)

    def log_pdf_multiple(self, X):
        return np.array([self.log_pdf(x) for x in X])
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        return 0.

    def get_parameter_names(self):
        return ['sigma', 'lmbda']
