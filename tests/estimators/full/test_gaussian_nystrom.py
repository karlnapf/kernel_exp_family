from nose.tools import assert_almost_equal
from numpy.testing.utils import assert_allclose

import kernel_exp_family.estimators.full.develop.gaussian as gaussian_full_develop
from kernel_exp_family.estimators.full.develop.gaussian_nystrom import log_pdf_naive,\
    grad_naive
from kernel_exp_family.estimators.full.gaussian import KernelExpFullGaussian
import kernel_exp_family.estimators.full.gaussian as gaussian_full
from kernel_exp_family.estimators.full.gaussian_nystrom import KernelExpFullNystromGaussian,\
    fit, log_pdf, grad
import numpy as np


def test_log_pdf_naive_nystrom_all_inds_equals_log_pdf_naive_full_alpha_equals_0():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    x = np.random.randn(D)
    sigma = 1.
    alpha = 0
    beta = np.random.randn(N, D)
    beta_nystrom = beta.flatten()
    inds = np.arange(N * D)
    
    a = gaussian_full_develop.log_pdf_naive(x, X, sigma, alpha, beta)
    b = log_pdf_naive(x, X, sigma, alpha, beta_nystrom, inds)
    
    assert_almost_equal(a, b)

def test_log_pdf_naive_nystrom_all_inds_equals_log_pdf_naive_full():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    x = np.random.randn(D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N, D)
    beta_nystrom = beta.flatten()
    inds = np.arange(N * D)
    
    a = gaussian_full_develop.log_pdf_naive(x, X, sigma, alpha, beta)
    b = log_pdf_naive(x, X, sigma, alpha, beta_nystrom, inds)
    
    assert_almost_equal(a, b)

def test_fit_nystrom_all_inds_equals_fit():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    lmbda = 0.001
    inds = np.arange(N * D)
    
    alpha_full, beta_full = gaussian_full.fit(X, sigma, lmbda)
    alpha, beta = fit(X, sigma, lmbda, inds)
    
    # low accuracy as numerically potentially unstable
    assert_allclose(alpha_full, alpha, rtol=1e-1)
    assert_allclose(beta_full.reshape(N*D,), beta, rtol=1e-1)

def test_full_pipeline_nystrom_all_inds_equals_full():
    sigma = 3
    lmbda = 0.01
    N = 10
    D = 2
    
    X = np.random.randn(N,D)
    X2 = np.random.randn(N,D)
    
    est = KernelExpFullNystromGaussian(sigma, lmbda, D, N, m=N*D)
    est.fit(X)
    est2 = KernelExpFullGaussian(sigma, lmbda, D, N)
    est2.fit(X)
    
    for x in X2:
        # low accuracy as numerically potentially unstable
        assert_allclose(est.log_pdf(x), est2.log_pdf(x), rtol=1e-2)

def test_grad_naive_nystrom_all_inds_equals_grad_naive_full_alpha_equals_0():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    alpha = 0.
    beta = np.random.randn(N, D)
    beta_nystrom = beta.flatten()
    inds = np.arange(N * D)
    
    for x in np.random.randn(N,D):
        a = gaussian_full_develop.grad_naive(x, X, sigma, alpha, beta)
        b = grad_naive(x, X, sigma, alpha, beta_nystrom, inds)
        assert_allclose(a, b)
        
        
def test_grad_naive_nystrom_all_inds_equals_grad_naive_full():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N, D)
    beta_nystrom = beta.flatten()
    inds = np.arange(N * D)
    
    for x in np.random.randn(N,D):
        a = gaussian_full_develop.grad_naive(x, X, sigma, alpha, beta)
        b = grad_naive(x, X, sigma, alpha, beta_nystrom, inds)
        assert_allclose(a, b)

def test_log_pdf_naive_equals_log_pdf():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N, D).flatten()
    inds = np.arange(N * D)
    
    for x in np.random.randn(N,D):
        a = log_pdf_naive(x, X, sigma, alpha, beta, inds)
        b = log_pdf(x, X, sigma, alpha, beta, inds)
        assert_allclose(a, b)
        
def test_grad_naive_equals_grad():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N, D).flatten()
    inds = np.arange(N * D)
    
    for x in np.random.randn(N,D):
        a = grad_naive(x, X, sigma, alpha, beta, inds)
        b = grad(x, X, sigma, alpha, beta, inds)
        assert_allclose(a, b)