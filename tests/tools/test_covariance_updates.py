from nose import SkipTest
from numpy.testing.utils import assert_allclose

from kernel_exp_family.tools.covariance_updates import update_mean_lmbda,\
    weights_to_lmbdas, update_mean_cov_L_lmbda, log_weights_to_lmbdas
import numpy as np


def test_update_mean_lmbda():
    N = 100
    D = 2
    X = np.random.randn(N, D)
    X2 = np.random.randn(N, D)
    
    lmbdas = np.ones(N) * 0.1
    old_mean = np.mean(X, axis=0)
    full_mean = update_mean_lmbda(X2, old_mean, lmbdas)
    for x, lmbda in zip(X2, lmbdas):
        old_mean = (1 - lmbda) * old_mean + lmbda * x
    
    assert_allclose(old_mean, full_mean)

def test_update_mean_cov_L_lmbda_converges_to_mean_and_cov():
    N_init = 10
    N = 10000
    D = 2
    X = np.random.randn(N, D)
    weights = np.ones(N)
    
    old_mean = np.mean(X[:N_init], axis=0)
    old_cov_L = np.linalg.cholesky(np.cov(X[:N_init].T, ddof=0))
    
    sum_old_weights = np.sum(weights[:N_init])
    lmbdas = weights_to_lmbdas(sum_old_weights, weights[N_init:])
    
    mean, cov_L = update_mean_cov_L_lmbda(X[N_init:], old_mean, old_cov_L, lmbdas)

    full_mean = np.mean(X, axis=0)
    
    # the above method uses N rather than N-1 to normalise covariance (biased)
    full_cov = np.cov(X.T, ddof=0)
    cov = np.dot(cov_L, cov_L.T)
    
    assert_allclose(full_mean, mean)
    assert_allclose(full_cov, cov, atol=1e-3)

def test_update_mean_cov_L_lmbda_converges_to_weighted_mean_and_cov():
    N_init = 10
    N = 10000
    D = 2
    X = np.random.randn(N, D)
    weights = np.random.rand(N)
    
    old_mean = np.average(X[:N_init], axis=0, weights=weights[:N_init])
    old_cov_L = np.linalg.cholesky(np.cov(X[:N_init].T, ddof=0))
    
    sum_old_weights = np.sum(weights[:N_init])
    lmbdas = weights_to_lmbdas(sum_old_weights, weights[N_init:])
    
    mean, cov_L = update_mean_cov_L_lmbda(X[N_init:], old_mean, old_cov_L, lmbdas)

    full_mean = np.average(X, axis=0, weights=weights)
    
    # the above method uses N rather than N-1 to normalise covariance (biased)
    try:
        full_cov = np.cov(X.T, ddof=0, aweights=weights)
    except TypeError:
        raise SkipTest("Numpy's cov method does not support aweights keyword.")
    
    cov = np.dot(cov_L, cov_L.T)
    
    assert_allclose(full_mean, mean)
    assert_allclose(full_cov, cov, atol=1e-2)

def test_weights_to_lmbdas_single():
    N = 1
    weights = np.ones(N)
    
    sum_old_weights = np.sum(weights)
    lmbdas = weights_to_lmbdas(sum_old_weights, weights)
    
    # if there are only two elements, the mean is produced by using lmbda = 0.5
    desired = np.array([0.5])
    assert_allclose(lmbdas, desired)

def test_weights_to_lmbdas_produces_mean():
    N = 30
    D = 2
    X = np.random.randn(N, D)
    
    full_mean_batch = np.mean(X, axis=0)
     
    sum_old_weights = 1
    new_weights = np.ones(N - 1)
    lmbdas = weights_to_lmbdas(sum_old_weights, new_weights)

    old_mean = X[0]
    full_mean = update_mean_lmbda(X[1:], old_mean, lmbdas)
     
    assert_allclose(full_mean_batch, full_mean)

def test_weights_to_lmbdas_produces_mean_weighted():
    N = 20
    X = np.random.randn(N)
    weights = np.random.rand(N)
    
    sum_old_weights = weights[0]
    lmbdas = weights_to_lmbdas(sum_old_weights, weights[1:])

    old_mean = X[0]
    full_mean = update_mean_lmbda(X[1:], old_mean, lmbdas)
    X_weighted = np.array([X[i] * weights[i] for i in range(N)])
    full_mean_batch = np.sum(X_weighted, axis=0) / np.sum(weights)
     
    assert_allclose(full_mean_batch, full_mean)

def test_weights_to_lmbdas_equals_log_version():
    N = 30
    
    sum_old_weights = 1
    new_weights = np.ones(N - 1)
    lmbdas = weights_to_lmbdas(sum_old_weights, new_weights)

    log_sum_old_weights = np.log(sum_old_weights)
    log_new_weights = np.log(new_weights)
    lmbdas2 = log_weights_to_lmbdas(log_sum_old_weights, log_new_weights)
     
    assert_allclose(lmbdas, lmbdas2)
