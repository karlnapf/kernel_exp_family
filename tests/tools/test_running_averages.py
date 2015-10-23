from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kernel_exp_family.tools.running_averages import online_mean_variance,\
    online_mean_covariance, rank_one_update_mean_covariance,\
    rank_m_update_mean_covariance,\
    rank_one_update_mean_covariance_cholesky_naive,\
    rank_one_update_mean_covariance_cholesky_lmbda
import numpy as np


def test_online_mean_variance():
    x = np.random.randn(100)
    
    mu_batch = np.mean(x)
    sigma2_batch = np.var(x, ddof=1)
    
    mu_online, sigma2_online = online_mean_variance(x)
    
    assert_close(mu_batch, mu_online)
    assert_close(sigma2_batch, sigma2_online)

def test_online_mean_covariance():
    D = 3
    X = np.random.randn(100, D)
    
    mu_batch = np.mean(X, 0)
    Sigma_batch = np.cov(X.T, ddof=1)
    
    mu_online, Sigma_online = online_mean_covariance(X)
    
    assert_close(mu_batch, mu_online)
    assert_close(Sigma_batch, Sigma_online)

def test_rank_one_update_mean_covariance():
    D = 3
    X = np.random.randn(100, D)
    
    mu_batch = np.mean(X, 0)
    Sigma_batch = np.cov(X.T, ddof=1)
    
    # first term
    mu_online = np.zeros(D)
    M2_online = np.zeros((D, D))
    n = 0
    
    for x in X:
        mu_online, Sigma_online, n, M2_online = rank_one_update_mean_covariance(x, n, mu_online, M2_online)
    
    assert_allclose(mu_batch, mu_online)
    assert_close(Sigma_batch, Sigma_online)

def test_rank_m_update_mean_covariance():
    D = 3
    X = np.random.randn(100, D)
    
    mu_batch = np.mean(X, 0)
    Sigma_batch = np.cov(X.T, ddof=1)
    
    # first term
    mu_online = np.zeros(D)
    M2_online = np.zeros((D, D))
    n = 0
    
    mu_online, Sigma_online, n, M2_online = rank_m_update_mean_covariance(X, n, mu_online, M2_online, ddof=1)
    
    assert_allclose(mu_batch, mu_online)
    assert_close(Sigma_batch, Sigma_online)

def test_rank_one_update_mean_covariance_cholesky_naive():
    D = 3
    N = 100
    N_half = N / 2
    X = np.random.randn(N, D)
    
    mu_batch = np.mean(X, 0)
    Sigma_batch = np.cov(X.T, ddof=1)
    Sigma_L_batch = np.linalg.cholesky(Sigma_batch)
    
    # first term
    mu_online = np.zeros(D)
    M2_online = np.zeros((D, D))
    n = 0
    
    # first terms without Cholesky
    for x in X[N_half:]:
        mu_online, _, n, M2_online = rank_one_update_mean_covariance(x, n, mu_online, M2_online)
    
    # first Cholesky of M2
    M2_L_online = np.linalg.cholesky(M2_online)
    
    # later terms are safe to compute Cholesky for
    for x in X[:N_half]:
        mu_online, Sigma_L_online, n, M2_online, M2_L_online = \
        rank_one_update_mean_covariance_cholesky_naive(x, n, mu_online, M2_online, M2_L_online)
    
    assert_allclose(mu_batch, mu_online)
    assert_close(Sigma_L_batch, Sigma_L_online)

def test_rank_one_update_mean_covariance_cholesky_lmbda():
    D = 3
    N = 100
    X = np.random.randn(N, D)
    
    mean = np.mean(X, 0)
    Sigma = np.cov(X.T)
    L = np.linalg.cholesky(Sigma)
    assert_allclose(np.dot(L, L.T), Sigma)
    
    # update with one more vector
    u = np.random.randn(D)
    lmbda = 0.1
    
    updated_mean = (1 - lmbda) * mean + lmbda * u
    updated_Sigma = (1 - lmbda) * Sigma + lmbda * np.outer(u - mean, u - mean)
    updated_L = np.linalg.cholesky(updated_Sigma)
    
    m_test, L_test = rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda, mean, L)
    assert_allclose(updated_mean, m_test)
    assert_allclose(updated_L, L_test)

def test_rank_one_update_mean_covariance_cholesky_lmbda_gamma2():
    D = 3
    N = 100
    X = np.random.randn(N, D)
    gamma2 = 2.
    
    mean = np.mean(X, 0)
    Sigma = np.cov(X.T)
    L = np.linalg.cholesky(Sigma)
    assert_allclose(np.dot(L, L.T), Sigma)
    
    # update with one more vector
    u = np.random.randn(D)
    lmbda = 0.1
    
    updated_mean = (1 - lmbda) * mean + lmbda * u
    updated_Sigma = (1 - lmbda) * Sigma + lmbda * np.outer(u - mean, u - mean) + lmbda * gamma2 * np.eye(D)
    updated_L = np.linalg.cholesky(updated_Sigma)
    
    m_test, L_test = rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda, mean, L, gamma2=gamma2)
    assert_allclose(updated_mean, m_test)
    assert_allclose(updated_L, L_test)

def test_rank_one_update_mean_covariance_cholesky_lmbda_nu2():
    D = 3
    N = 100
    X = np.random.randn(N, D)
    nu2 = 2.
    
    mean = np.mean(X, 0)
    Sigma = np.cov(X.T)
    L = np.linalg.cholesky(Sigma)
    assert_allclose(np.dot(L, L.T), Sigma)
    
    # update with one more vector
    u = np.random.randn(D)
    lmbda = 0.1
    
    updated_mean = (1 - lmbda) * mean + lmbda * u
    updated_Sigma = (1 - lmbda) * Sigma + lmbda * nu2 * np.outer(u - mean, u - mean)
    updated_L = np.linalg.cholesky(updated_Sigma)
    
    m_test, L_test = rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda, mean, L, nu2=2.)
    assert_allclose(updated_mean, m_test)
    assert_allclose(updated_L, L_test)
