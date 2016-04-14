from kernel_exp_family.estimators.full.gaussian import SE_dx_dx, SE_dx,\
    build_system_even_faster
import numpy as np


def nystrom_naive(X, sigma, lmbda, inds):
    A, b = build_system_even_faster(X, sigma, lmbda)
    
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
    
    l = np.sqrt(np.float(sigma) / 2)
    SE_dx_dx_l = lambda x, y : SE_dx_dx(x, y, l)
    SE_dx_l = lambda x, y: SE_dx(x, y, l)
    
    xi = 0
    betasum = 0
    
    ais = [ind_to_ai(ind, D) for ind in range(len(inds))]
    
    for ind, (a,i) in enumerate(ais):
        x_a = X[a, :].reshape(-1, 1)
        gradient_x_xa = np.squeeze(SE_dx_l(x.reshape(-1, 1), x_a))
        xi_grad = SE_dx_dx_l(x.reshape(-1, 1), x_a)
        
        xi += xi_grad[i] / N
        betasum += gradient_x_xa[i] * beta[ind]
    
    return np.float(alpha * xi + betasum)
