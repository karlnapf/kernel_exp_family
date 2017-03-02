from nose.tools import assert_almost_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kernel_exp_family.estimators.full.develop.gaussian import build_system_loop, \
    compute_lower_right_submatrix_loop, compute_RHS_loop, log_pdf_naive, \
    grad_naive, compute_h_old_interface
from kernel_exp_family.estimators.full.develop.gaussian_depracated import build_system, \
    compute_h, log_pdf, \
    second_order_grad, compute_objective, grad, fit, compute_xi_norm_2
from kernel_exp_family.kernels.develop.kernels import SE_dx_dy, SE_dx_dx_dy
import numpy as np


def setup(N=10, D=3):
    """ Generates some data and parameters """
    print N
    sigma = np.random.randn() ** 2
    l = np.sqrt(np.float(sigma) / 2)
    lmbda = np.random.randn() ** 2

    mean = np.random.randn(D)
    cov = np.random.rand(D, D)
    cov = np.dot(cov, cov.T)
    data = np.random.multivariate_normal(mean, cov, size=10)

    return data, l, sigma, lmbda


def test_build_system_loop_equals_implementation():
    data, _, sigma, lmbda = setup()

    A_new, b_new = build_system(data, data, sigma, lmbda)

    A_old, b_old = build_system_loop(data, sigma, lmbda)

    assert_close(A_new, A_old, verbose=True)
    assert_close(b_new, np.squeeze(b_old.T))


def test_compute_lower_submatrix():
    data, l, sigma, lmbda = setup()

    kernel_dx_dy = lambda x, y: SE_dx_dy(x, y, l)

    A_loop = compute_lower_right_submatrix_loop(kernel_dx_dy, data, lmbda)

    A, _ = build_system(data, data, sigma, lmbda)
    A_vector = A[1:, 1:]
#     A_vector = compute_lower_right_submatrix(all_hessians, data.shape[0], lmbda)

    assert_close(A_loop, A_vector)

def test_compute_rhs_vector():
    data, l, sigma, lmbda = setup()

    xi_norm_2 = compute_xi_norm_2(data, data, sigma)

    kernel_dx_dx_dy = lambda x, y: SE_dx_dx_dy(x, y, l)

    _, rhs_vector = build_system(data, data, sigma, lmbda)
    rhs_loop = compute_RHS_loop(kernel_dx_dx_dy, data, xi_norm_2)

    assert_close(rhs_vector, np.squeeze(rhs_loop.T))

def test_log_pdf_equals_log_pdf_naive():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N, D)
    
    for x in np.random.randn(N, D):
        a = log_pdf(x, X, sigma, alpha, beta)
        b = log_pdf_naive(x, X, sigma, alpha, beta)
        assert_almost_equal(a, b)

def test_grad_equals_grad_naive():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N, D)
    
    for x in np.random.randn(N, D):
        a = grad(x, X, sigma, alpha, beta)
        b = grad_naive(x, X, sigma, alpha, beta)
        assert_allclose(a, b)

def test_gradient_execute():
    N = 10
    D = 2
    X = np.random.randn(N, D)
    x = np.random.randn(D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N, D)
    
    grad(x, X, sigma, alpha, beta)

def test_compute_h_equals_old_interface():
    data, l, sigma, _ = setup()

    kernel_dx_dx_dy = lambda x, y: SE_dx_dx_dy(x, y, l)

    reference = compute_h_old_interface(kernel_dx_dx_dy, data).reshape(-1)
    implementation = compute_h(data, data, sigma)
    
    assert_close(reference, implementation)


def test_second_order_grad_execute():
    data, _, sigma, _ = setup()
    N, D = data.shape
    x = np.random.randn(D)

    alpha = np.random.randn()
    beta = np.random.randn(N, D)

    second_order_grad(x, data, sigma, alpha, beta)


def test_compute_objective_execute():
    X_train, _, sigma, _ = setup(10, 3)
    X_test, _, _, _ = setup(5, 3)

    N, D = X_train.shape

    alpha = np.random.randn()
    beta = np.random.randn(N, D)

    compute_objective(X_test, X_train, sigma, alpha, beta)

def test_build_system_custom_basis_execute():
    data, _, sigma, lmbda = setup()
    basis = np.random.randn(10, data.shape[1])
    build_system(basis, data, sigma, lmbda)

def test_fit_custom_basis_execute():
    data, _, sigma, lmbda = setup()
    basis = np.random.randn(10, data.shape[1])
    fit(basis, data, sigma, lmbda)

def test_fit_custom_basis_equals_full_execute():
    data, _, sigma, lmbda = setup()
    basis = data.copy()
    alpha, beta = fit(basis, data, sigma, lmbda)
    alpha2, beta2 = fit(basis, data, sigma, lmbda)
    
    assert_close(alpha, alpha2)
    assert_close(beta, beta2)
    
