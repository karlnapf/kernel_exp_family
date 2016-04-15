from kernel_exp_family.estimators.full.gaussian import build_system
from kernel_exp_family.kernels.kernels import gaussian_kernel_dx_dx,\
    gaussian_kernel_grad, gaussian_kernel_dx_i_dx_i_dx_j,\
    gaussian_kernel_dx_i_dx_j
import numpy as np


def nystrom_naive(X, sigma, lmbda, inds):
    A, b = build_system(X, sigma, lmbda)
    
    inds_with_xi = np.zeros(len(inds)+1)
    inds_with_xi[1:] = (inds+1)
    inds_with_xi = inds_with_xi.astype(np.int)
    
    A_nm = A[:, inds_with_xi]
    
    return A_nm, b

def ind_to_ai(ind, D):
    """
    For a given row index of the A matrix, return corresponding data and component index
    """
    return ind/D, ind%D

def log_pdf_naive(x, X, sigma, alpha, beta, inds):
    N, D = X.shape
    
    xi = 0
    betasum = 0
    
    ais = [ind_to_ai(ind, D) for ind in range(len(inds))]
    
    for ind, (a,i) in enumerate(ais):
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
    
    for ind, (a,i) in enumerate(ais):
        x_a = X[a]
        xi_gradient_mat = gaussian_kernel_dx_i_dx_i_dx_j(x, x_a, sigma)
        left_arg_hessian = gaussian_kernel_dx_i_dx_j(x, x_a, sigma)
        xi_grad += xi_gradient_mat[i] / N
        betasum_grad += beta[ind] * left_arg_hessian[i]

    return alpha * xi_grad + betasum_grad
