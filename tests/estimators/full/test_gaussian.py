from nose.tools import assert_almost_equal
from numpy.ma.testutils import assert_close
import numpy as np

from kernel_exp_family.estimators.full.develop.gaussian import compute_lower_right_submatrix_loop, \
    compute_RHS_loop, log_pdf_naive, build_system_loop, compute_h_old_interface
from kernel_exp_family.estimators.full.gaussian import KernelExpFullGaussian, build_system, \
    compute_RHS, log_pdf, compute_lower_right_submatrix,\
    compute_h, grad
from kernel_exp_family.kernels.kernels import gaussian_kernel_hessians
from kernel_exp_family.kernels.develop.kernels import SE_dx_i_dx_j, \
    SE_dx_i_dx_i_dx_j, SE, SE_dx, SE_dx_dy, SE_dx_dx_dy

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


def test_build_system_loop_equals_implementation():
    data, _, sigma, lmbda  = setup()

    A_new, b_new = build_system(data, sigma, lmbda)

    A_old, b_old = build_system_loop(data, sigma, lmbda)

    assert_close(A_new, A_old, verbose=True)
    assert_close(b_new, np.squeeze(b_old.T))


def test_compute_lower_submatrix():
    data, l, sigma, lmbda  = setup()

    kernel_dx_dy = lambda x,y: SE_dx_dy(x, y, l)

    A_loop = compute_lower_right_submatrix_loop(kernel_dx_dy, data, lmbda)

    all_hessians = gaussian_kernel_hessians(data, sigma=sigma)
    A_vector = compute_lower_right_submatrix(all_hessians, data.shape[0], lmbda)

    assert_close(A_loop, A_vector)

def test_compute_RHS_vector():
    data, l, sigma, _  = setup()

    xi_norm_2 = np.random.randn()

    kernel_dx_dx_dy = lambda x,y: SE_dx_dx_dy(x,y,l)

    h = compute_h(data, sigma)
    rhs_vector = compute_RHS(h, xi_norm_2)
    rhs_loop = compute_RHS_loop(kernel_dx_dx_dy, data, xi_norm_2)

    assert_close(rhs_vector, np.squeeze(rhs_loop.T))


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

def test_gradient_execute():
    N=10
    D=2
    X = np.random.randn(N,D)
    x = np.random.randn(D)
    sigma = 1.
    alpha = np.random.randn()
    beta = np.random.randn(N,D)
    
    grad(x, X, sigma, alpha, beta)

def test_compute_h_equals_old_interface():
    data, l, sigma, _  = setup()

    kernel_dx_dx_dy = lambda x,y: SE_dx_dx_dy(x,y,l)

    reference = compute_h_old_interface(kernel_dx_dx_dy, data)
    implementation = compute_h(data, sigma)
    
    assert_close(reference, implementation)