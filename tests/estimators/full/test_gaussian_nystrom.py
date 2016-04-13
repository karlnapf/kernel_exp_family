from nose.tools import assert_almost_equal

import kernel_exp_family.estimators.full.develop.gaussian as gaussian_full
from kernel_exp_family.estimators.full.develop.gaussian_nystrom import log_pdf_naive
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
    
    a = gaussian_full.log_pdf_naive(x, X, sigma, alpha, beta)
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
    
    a = gaussian_full.log_pdf_naive(x, X, sigma, alpha, beta)
    b = log_pdf_naive(x, X, sigma, alpha, beta_nystrom, inds)
    
    assert_almost_equal(a, b)
