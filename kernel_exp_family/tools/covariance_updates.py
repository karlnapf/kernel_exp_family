from choldate._choldate import cholupdate

from kernel_exp_family.tools.numerics import log_sum_exp
import numpy as np


def update_mean_lmbda(X, old_mean, lmbdas):
    assert len(X) == len(lmbdas)
    
    mean = old_mean
    for x, lmbda in zip(X, lmbdas):
        mean = (1 - lmbda) * mean + lmbda * x
    
    return mean

def update_mean_cov_L_lmbda(X, old_mean, old_cov_L, lmbdas):
    assert len(X) == len(lmbdas)
    
    # work on upper triangular cholesky internally
    old_cov_R = old_cov_L.T
    
    mean = old_mean
    for x, lmbda in zip(X, lmbdas):
        old_cov_R *= np.sqrt(1 - lmbda)
        update_vec = np.sqrt(lmbda) * (x - mean)
        cholupdate(old_cov_R, update_vec)
        mean = (1 - lmbda) * mean + lmbda * x
    
    # transform back to lower triangular version
    cov_L = old_cov_R.T
    
    return mean, cov_L

def weights_to_lmbdas(sum_old_weights, new_weights, boundary_check_min_number=1e-5):
    N = len(new_weights)
    lmbdas = np.zeros(N)
    
    for i, new_weight in enumerate(new_weights):
        sum_old_weights += new_weight
        lmbdas[i] = new_weight / (sum_old_weights)
    
    # numerical checks for lambdas. Must be in (0,1)
    lmbdas[lmbdas < boundary_check_min_number] = boundary_check_min_number
    lmbdas[(1 - lmbdas) < boundary_check_min_number] = 1 - boundary_check_min_number
    
    return lmbdas

def log_weights_to_lmbdas(log_sum_old_weights, log_new_weights, boundary_check_min_number=1e-5):
    N = len(log_new_weights)
    lmbdas = np.zeros(N)
    
    for i, log_new_weight in enumerate(log_new_weights):
        log_sum_old_weights = log_sum_exp([log_sum_old_weights, log_new_weight])
        log_lmbda = log_new_weight - log_sum_old_weights
        lmbdas[i] = np.exp(log_lmbda)
    
    # numerical checks for lambdas. Must be in (0,1)
    lmbdas[lmbdas < boundary_check_min_number] = boundary_check_min_number
    lmbdas[(1 - lmbdas) < boundary_check_min_number] = 1 - boundary_check_min_number
    
    return lmbdas
