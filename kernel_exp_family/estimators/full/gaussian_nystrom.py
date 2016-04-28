from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.estimators.full.develop.gaussian_nystrom import ind_to_ai
from kernel_exp_family.estimators.full.gaussian import compute_h, \
    compute_xi_norm_2, compute_RHS
from kernel_exp_family.kernels.kernels import gaussian_kernel_dx_component, \
    gaussian_kernel_dx_dx_component, gaussian_kernel_dx_i_dx_i_dx_j_component, \
    gaussian_kernel_dx_i_dx_j_component, gaussian_kernel_hessian_entry
from kernel_exp_family.tools.assertions import assert_array_shape
import numpy as np


def compute_first_row_without_storing(X, h, n, lmbda, sigma):
    N_x, d = X.shape
    result = np.zeros(h.shape)
    for ind1 in range(len(result)):
        a, i = ind_to_ai(ind1, d)
        for ind2 in range(N_x * d):
            b, j = ind_to_ai(ind2, d)
            H = gaussian_kernel_hessian_entry(X[a], X[b], i, j, sigma)
            result[ind1] += h[ind2] * H
    result /= n
    result += lmbda * h
    
    return result

def compute_lower_right_submatrix_component(data, lmbda, idx1, idx2, sigma):
    n, d = data.shape

    a, i = ind_to_ai(idx1, d)
    b, j = ind_to_ai(idx2, d)
    x_a = data[a]
    x_b = data[b]
    G_a_b_i_j = gaussian_kernel_hessian_entry(x_a, x_b, i, j, sigma)
    
    G_sum = 0.
    for idx_n in range(n):
        x_n = data[idx_n]
        for idx_d in range(d):
            G1 = gaussian_kernel_hessian_entry(x_a, x_n, i, idx_d, sigma)
            G2 = gaussian_kernel_hessian_entry(x_n, x_b, idx_d, j, sigma)
            G_sum += G1 * G2

    return G_sum / n + lmbda * G_a_b_i_j

def build_system_nystrom(X, sigma, lmbda, inds):
    N, D = X.shape
    m = len(inds)

    h = compute_h(X, sigma).reshape(-1)
    xi_norm_2 = compute_xi_norm_2(X, sigma)
    
    A_mn = np.zeros((m + 1, N * D + 1))
    A_mn[0, 0] = np.dot(h, h) / N + lmbda * xi_norm_2
    
    for row_idx in range(len(inds)):
        for col_idx in range(N * D):
            A_mn[1 + row_idx, 1 + col_idx] = compute_lower_right_submatrix_component(X, lmbda, inds[row_idx], col_idx, sigma)
    
    A_mn[0, 1:] = compute_first_row_without_storing(X, h, N, lmbda, sigma)
    A_mn[1:, 0] = A_mn[0, inds + 1]
    
    b = compute_RHS(h, xi_norm_2)
    
    return A_mn, b

def fit(X, sigma, lmbda, inds):
    A_mn, b = build_system_nystrom(X, sigma, lmbda, inds)
    
    A = np.dot(A_mn, A_mn.T)
    b = np.dot(A_mn, b).flatten()
    
    # x = np.linalg.solve(A, b)
    # pseudo-inverse calculation via eigendecomposition
    x = np.dot(np.linalg.pinv(A), b)
    
    alpha = x[0]
    beta = x[1:]
    return alpha, beta

def log_pdf(x, X, sigma, alpha, beta, inds):
    N, D = X.shape
    
    xi = 0
    betasum = 0
    
    ais = [ind_to_ai(ind, D) for ind in range(len(inds))]
    
    for ind, (a, i) in enumerate(ais):
        gradient_x_xa_i = gaussian_kernel_dx_component(x, X[a], i, sigma)
        xi_grad_i = gaussian_kernel_dx_dx_component(x, X[a], i, sigma)
        
        xi += xi_grad_i / N
        betasum += gradient_x_xa_i * beta[ind]
    
    return np.float(alpha * xi + betasum)

def grad(x, X, sigma, alpha, beta, inds):
    N, D = X.shape
    
    xi_grad = 0
    betasum_grad = 0
    
    ais = [ind_to_ai(ind, D) for ind in range(len(inds))]
    
    for ind, (a, i) in enumerate(ais):
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
