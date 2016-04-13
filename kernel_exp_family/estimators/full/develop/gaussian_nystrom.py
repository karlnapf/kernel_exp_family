from kernel_exp_family.estimators.full.gaussian import SE_dx_dx, SE_dx
import numpy as np
from kernel_exp_family.estimators.full.gaussian_nystrom import ind_to_ai


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
    
    return alpha * xi + betasum
