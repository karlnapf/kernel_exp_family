from autograd import hessian, grad
from nose.tools import assert_almost_equal
from numpy.ma.testutils import assert_close

import autograd.numpy as np  # Thinly-wrapped numpy
from kernel_exp_family.estimators.full.develop.gaussian import compute_lower_right_submatrix_loop, \
    compute_RHS_loop, log_pdf_naive
from kernel_exp_family.estimators.full.gaussian import SE_dx_i_dx_j, \
    SE_dx_i_dx_i_dx_j, SE, SE_dx, KernelExpFullGaussian, build_system, \
    build_system_fast, SE_dx_dy, compute_lower_right_submatrix, compute_RHS, \
    SE_dx_dx_dy, build_system_even_faster, log_pdf


def setup():
    """ Generates some data and parameters """
    sigma = np.random.randn()**2
    l = np.sqrt(np.float(sigma) / 2)
    lmbda = np.random.randn()**2
    N = 10
    D = 2

    mean = np.random.randn(D)
    cov = np.random.rand(D,D)
    cov = np.dot(cov,cov.T)

    data = np.random.multivariate_normal(mean, cov, size=N)

    return data, l, sigma, lmbda


def test_SE_dx_i_dx_j():
    d = 3
    l = 2.0

    x = np.random.randn(d).reshape(-1, 1)
    y = np.random.randn(d).reshape(-1, 1)

    left_arg_hessian = SE_dx_i_dx_j(x, y, l)
    autograd_left_arg_hessian = hessian(lambda a: SE(a, y, l))
    autograd_computation = np.squeeze(autograd_left_arg_hessian(x))

    assert_close(left_arg_hessian, autograd_computation)


def test_SE_dx_i_dx_i_dx_j():
    d = 3
    l = 2.0

    x = np.random.randn(d).reshape(-1, 1)
    y = np.random.randn(d).reshape(-1, 1)

    derivative_matrix = SE_dx_i_dx_i_dx_j(x, y, l)
    autograd_derivative_matrix = hessian(lambda a: SE_dx(a, y, l))
    autograd_computation = np.squeeze(autograd_derivative_matrix(x))

    d = x.shape[0]
    autograd_result = np.zeros((d, d))

    for i in range(d):
        autograd_result[i, :] = autograd_computation[i, i, :]

    assert_close(derivative_matrix, autograd_result)

def test_grad():
    sigma = 1.
    lmbda = 1.
    N = 10
    D = 2
    est = KernelExpFullGaussian(sigma, lmbda, D, N)

    X = np.random.randn(N, D)
    est.fit(X)

    auto_gradient = grad(est.log_pdf)

    x_new = np.random.randn(D)

    # print(est.grad(x_new))
    # print(auto_gradient(x_new))

    assert_close(est.grad(x_new), auto_gradient(x_new))

def test_build_system_old_new():
    data, _, sigma, lmbda  = setup()

    A_new, b_new = build_system_fast(data, sigma, lmbda)

    A_old, b_old = build_system(data, sigma, lmbda)

    assert_close(A_new, A_old, verbose=True)
    assert_close(b_new, b_old)

def test_compute_lower_submatrix():
    data, l, _, lmbda  = setup()

    kernel_dx_dy = lambda x,y: SE_dx_dy(x, y, l)

    A_loop = compute_lower_right_submatrix_loop(kernel_dx_dy, data, lmbda)
    A_vector = compute_lower_right_submatrix(kernel_dx_dy, data, lmbda)

    assert_close(A_loop, A_vector)

def test_compute_RHS_vector():
    data, l, _, _  = setup()

    xi_norm_2 = np.random.randn()

    kernel_dx_dx_dy = lambda x,y: SE_dx_dx_dy(x,y,l)

    rhs_vector = compute_RHS(kernel_dx_dx_dy, data, xi_norm_2)
    rhs_loop = compute_RHS_loop(kernel_dx_dx_dy, data, xi_norm_2)

    assert_close(rhs_vector, rhs_loop)

def test_build_system_even_fast():
    data, _, sigma, lmbda  = setup()
    
    A_new, b_new = build_system_even_faster(data, sigma, lmbda)

    A_old, b_old = build_system_fast(data, sigma, lmbda)

    assert_close(A_new, A_old, verbose=True)
    assert_close(b_new, np.squeeze(b_old.T))

def test_log_pdf_equals_log_pdf_naive():
    N=10
    D=2
    X = np.random.randn(N,D)
    x = np.random.randn(D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N,D)
    
    a = log_pdf(x, X, sigma, alpha, beta)
    b = log_pdf_naive(x, X, sigma, alpha, beta)
    
    assert_almost_equal(a,b)