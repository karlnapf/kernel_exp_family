from kernel_exp_family.estimators.full.gaussian import build_system, compute_h, \
    compute_xi_norm_2, compute_first_row, compute_RHS
from kernel_exp_family.estimators.full.gaussian_nystrom import compute_lower_right_submatrix_component,\
    compute_first_row_without_storing
from kernel_exp_family.kernels.kernels import gaussian_kernel_dx_dx, \
    gaussian_kernel_grad, gaussian_kernel_dx_i_dx_i_dx_j, \
    gaussian_kernel_dx_i_dx_j, gaussian_kernel_hessians
import numpy as np

def ind_to_ai(ind, D):
    """
    For a given row index of the A matrix, return corresponding data and component index
    """
    return ind / D, ind % D

def build_system_nystrom_modular_slow(X, sigma, lmbda, inds):
    N, D = X.shape
    m = len(inds)

    h = compute_h(X, sigma).reshape(-1)
    xi_norm_2 = compute_xi_norm_2(X, sigma)
    
    A_mn = np.zeros((m + 1, N * D + 1))
    A_mn[0, 0] = np.dot(h, h) / N + lmbda * xi_norm_2
    
    for row_idx in range(len(inds)):
        for col_idx in range(N * D):
            A_mn[1 + row_idx, 1 + col_idx] = compute_lower_right_submatrix_component(X, lmbda, inds[row_idx], col_idx, sigma)
    
    ## compute in parallel with joblib
    #results = Parallel(n_jobs=4)(delayed(compute_lower_right_submatrix_component)(X, lmbda, inds[row_idx], col_idx, sigma) for (row_idx, col_idx) in row_col_ind_pairs)
    
    
    A_mn[0, 1:] = compute_first_row_without_storing(X, h, N, lmbda, sigma)
    A_mn[1:, 0] = A_mn[0, inds + 1]
    
    b = compute_RHS(h, xi_norm_2)
    
    return A_mn, b

def build_system_nystrom_naive_from_all_hessians(X, sigma, lmbda, inds):
    n, d = X.shape
    m = len(inds)

    h = compute_h(X, sigma).reshape(-1)
    all_hessians = gaussian_kernel_hessians(X, sigma=sigma)
    xi_norm_2 = compute_xi_norm_2(X, sigma)
    
    A_mn = np.zeros((m + 1, n * d + 1))
    A_mn[0, 0] = np.dot(h, h) / n + lmbda * xi_norm_2
    
    G_nm = np.dot(all_hessians[inds, :], all_hessians) / n + lmbda * all_hessians[inds, :]
    A_mn[1:, 1:] = G_nm
    
    A_mn[0, 1:] = compute_first_row(h, all_hessians, n, lmbda)
    A_mn[1:, 0] = A_mn[0, inds + 1]
    
    b = compute_RHS(h, xi_norm_2)
    
    return A_mn, b

def build_system_nystrom_naive_from_full(X, sigma, lmbda, inds):
    A, b = build_system(X, sigma, lmbda)
    
    inds_with_xi = np.zeros(len(inds) + 1)
    inds_with_xi[1:] = (inds + 1)
    inds_with_xi = inds_with_xi.astype(np.int)
    
    A_nm = A[:, inds_with_xi]
    
    return A_nm, b

def log_pdf_naive(x, X, sigma, alpha, beta, inds):
    N, D = X.shape
    
    xi = 0
    betasum = 0
    
    ais = [ind_to_ai(ind, D) for ind in range(len(inds))]
    
    for ind, (a, i) in enumerate(ais):
        x_a = np.atleast_2d(X[a, :])
        gradient_x_xa = np.squeeze(gaussian_kernel_grad(x, x_a, sigma))
        xi_grad = np.squeeze(gaussian_kernel_dx_dx(x, x_a, sigma))
        
        xi += xi_grad[i] / N
        betasum += gradient_x_xa[i] * beta[ind]
    
    return np.float(alpha * xi + betasum)

def grad_naive(x, X, sigma, alpha, beta, inds):
    N, D = X.shape
    
    xi_grad = 0
    betasum_grad = 0
    
    ais = [ind_to_ai(ind, D) for ind in range(len(inds))]
    
    for ind, (a, i) in enumerate(ais):
        x_a = X[a]
        xi_gradient_mat = gaussian_kernel_dx_i_dx_i_dx_j(x, x_a, sigma)
        left_arg_hessian = gaussian_kernel_dx_i_dx_j(x, x_a, sigma)
        xi_grad += xi_gradient_mat[i] / N
        betasum_grad += beta[ind] * left_arg_hessian[i]

    return alpha * xi_grad + betasum_grad
