from nose.tools import assert_almost_equal
from numpy.testing.utils import assert_allclose

import kernel_exp_family.estimators.full.develop.gaussian as gaussian_full_develop
from kernel_exp_family.estimators.full.develop.gaussian_nystrom import log_pdf_naive, \
    grad_naive, build_system_nystrom_naive_from_full, \
    build_system_nystrom_naive_from_all_hessians,\
    build_system_nystrom_modular_slow
from kernel_exp_family.estimators.full.gaussian import KernelExpFullGaussian, \
    compute_lower_right_submatrix, compute_first_row, compute_h
import kernel_exp_family.estimators.full.gaussian as gaussian_full
from kernel_exp_family.estimators.full.gaussian_nystrom import KernelExpFullNystromGaussian, \
    fit, log_pdf, grad, build_system_nystrom, \
    compute_lower_right_submatrix_component, compute_first_row_without_storing
from kernel_exp_family.kernels.kernels import gaussian_kernel_hessians
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
    assert_allclose(beta_full.reshape(N * D,), beta, rtol=1e-1)

def test_full_pipeline_nystrom_all_inds_equals_full():
    sigma = 3
    lmbda = 0.01
    N = 10
    D = 2
    
    X = np.random.randn(N, D)
    X2 = np.random.randn(N, D)
    
    est = KernelExpFullNystromGaussian(sigma, lmbda, D, N, m=N * D)
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
    
    for x in np.random.randn(N, D):
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
    
    for x in np.random.randn(N, D):
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
    
    for x in np.random.randn(N, D):
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
    
    for x in np.random.randn(N, D):
        a = grad_naive(x, X, sigma, alpha, beta, inds)
        b = grad(x, X, sigma, alpha, beta, inds)
        assert_allclose(a, b)

def test_build_system_nystrom_equals_build_system_nystrom_naive_from_full():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    lmbda = 0.1
    inds = np.arange(N * D)
    
    A, b = build_system_nystrom(X, sigma, lmbda, inds)
    A_naive, b_naive = build_system_nystrom_naive_from_full(X, sigma, lmbda, inds)
    
    assert_allclose(A, A_naive)
    assert_allclose(b, b_naive)

def test_build_system_nystrom_naive_from_all_hessians_equals_build_system_nystrom_naive_from_full():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    lmbda = 0.1
    inds = np.arange(N * D)
    
    A, b = build_system_nystrom_naive_from_all_hessians(X, sigma, lmbda, inds)
    A_naive, b_naive = build_system_nystrom_naive_from_full(X, sigma, lmbda, inds)
    
    assert_allclose(A, A_naive)
    assert_allclose(b, b_naive)

def test_build_system_nystrom_equals_build_system_nystrom_naive_from_all_hessians():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    lmbda = 0.1
    inds = np.arange(N * D)
    
    A, b = build_system_nystrom(X, sigma, lmbda, inds)
    A_naive, b_naive = build_system_nystrom_naive_from_all_hessians(X, sigma, lmbda, inds)
    
    assert_allclose(A, A_naive)
    assert_allclose(b, b_naive)

def test_compute_lower_right_submatrix_component_equals_compute_lower_right_submatrix():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    lmbda = 0.1
    all_hessians = gaussian_kernel_hessians(X, sigma=sigma)
    A = compute_lower_right_submatrix(all_hessians, N, lmbda)
    
    for idx1 in range(N * D):
        for idx2 in range(N * D):
            A_component = compute_lower_right_submatrix_component(X, lmbda, idx1, idx2, sigma)
            assert_allclose(A[idx1, idx2], A_component)

def test_compute_first_row_without_storing_equals_compute_first_row():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    lmbda = 0.1
    all_hessians = gaussian_kernel_hessians(X, sigma=sigma)
    h = compute_h(X, sigma).reshape(-1)
    row = compute_first_row(h, all_hessians, N, lmbda)
    row2 = compute_first_row_without_storing(X, h, N, lmbda, sigma)
    
    assert_allclose(row, row2)
    
def test_build_system_nystrom_modular_slow_equals_build_system_nystrom():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    lmbda = 0.1
    inds = np.arange(N * D)
    
    A, b = build_system_nystrom(X, sigma, lmbda, inds)
    A_naive, b_naive = build_system_nystrom_modular_slow(X, sigma, lmbda, inds)
    
    assert_allclose(A, A_naive)
    assert_allclose(b, b_naive)
