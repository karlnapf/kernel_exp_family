from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kernel_exp_family.estimators.full.gaussian import fit, log_pdf, grad, \
    second_order_grad, compute_objective
import kernel_exp_family.estimators.full.develop.gaussian_depracated as gaussian_depracated
import numpy as np


def setup(N=10, D=3):
    """ Generates some data and parameters """
    sigma = np.random.randn() ** 2
    l = np.sqrt(np.float(sigma) / 2)
    lmbda = np.random.randn() ** 2

    mean = np.random.randn(D)
    cov = np.random.rand(D, D)
    cov = np.dot(cov, cov.T)
    data = np.random.multivariate_normal(mean, cov, size=10)

    return data, l, sigma, lmbda

def test_fit_beta_equals_deprecated():
    data, _, sigma, lmbda = setup()
    basis = data.copy()
    beta = fit(basis, data, sigma, lmbda)
    _, beta_deprecated = gaussian_depracated.fit(basis, data, sigma, lmbda)
    
    assert_allclose(beta, beta_deprecated, rtol=1e-06, atol=1e-5)

def test_log_pdf_equals_deprecated():
    data, _, sigma, lmbda = setup()
    basis = data.copy()
    beta = fit(basis, data, sigma, lmbda)
    alpha_deprecated, beta_deprecated = gaussian_depracated.fit(basis, data, sigma, lmbda)
    
    for x in data:
        lpdf = log_pdf(x, basis, sigma, lmbda, beta)
        lpdf_deprecated = gaussian_depracated.log_pdf(x, basis, sigma, alpha_deprecated, beta_deprecated)
        assert_close(lpdf, lpdf_deprecated)

def test_grad_equals_deprecated():
    data, _, sigma, lmbda = setup()
    basis = data.copy()
    beta = fit(basis, data, sigma, lmbda)
    alpha_deprecated, beta_deprecated = gaussian_depracated.fit(basis, data, sigma, lmbda)
    
    for x in data:
        g = grad(x, basis, sigma, lmbda, beta)
        g_deprecated = gaussian_depracated.grad(x, basis, sigma, alpha_deprecated, beta_deprecated)
        assert_allclose(g, g_deprecated, rtol=1e-06, atol=1e-5)

def test_second_order_grad_equals_deprecated():
    data, _, sigma, lmbda = setup()
    basis = data.copy()
    beta = fit(basis, data, sigma, lmbda)
    alpha_deprecated, beta_deprecated = gaussian_depracated.fit(basis, data, sigma, lmbda)
    
    for x in data:
        g2 = second_order_grad(x, basis, sigma, lmbda, beta)
        g2_deprecated = gaussian_depracated.second_order_grad(x, basis, sigma, alpha_deprecated, beta_deprecated)
        assert_allclose(g2, g2_deprecated, rtol=1e-06, atol=1e-5)

def test_compute_objective_equals_deprecated():
    data, _, sigma, lmbda = setup()
    basis = data.copy()
    beta = fit(basis, data, sigma, lmbda)
    alpha_deprecated, beta_deprecated = gaussian_depracated.fit(basis, data, sigma, lmbda)
    
    o = compute_objective(data, basis, sigma, lmbda, beta)
    o_deprecated = gaussian_depracated.compute_objective(data, basis, sigma, alpha_deprecated, beta_deprecated)
    assert_close(o, o_deprecated)

def test_fit_nystrom_execute():
    data, _, sigma, lmbda = setup()
    inds = np.random.permutation(data.shape[0])[:data.shape[0] / 2]
    basis = data[inds].copy()
    
    fit(basis, data, sigma, lmbda)

def test_log_pdf_grad_second_order_grad_objective_nystrom_execute():
    data, _, sigma, lmbda = setup()
    inds = np.random.permutation(data.shape[0])[:data.shape[0] / 2]
    basis = data[inds].copy()
    
    beta = fit(basis, data, sigma, lmbda)
    for x in data:
        log_pdf(x, basis, sigma, lmbda, beta)
        grad(x, basis, sigma, lmbda, beta)
        second_order_grad(x, basis, sigma, lmbda, beta)
    
    compute_objective(data, basis, sigma, lmbda, beta)
