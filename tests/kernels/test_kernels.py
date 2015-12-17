from nose import SkipTest
from nose.tools import assert_almost_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kernel_exp_family.kernels.develop.kernels import rff_feature_map_grad_loop,\
    rff_feature_map_grad2_loop
from kernel_exp_family.kernels.kernels import gaussian_kernel_theano, \
    gaussian_kernel_grad_theano, gaussian_kernel_hessian_theano, \
    theano_available, gaussian_kernel_third_order_derivative_tensor_theano, gaussian_kernel, \
    gaussian_kernel_grad, rff_feature_map_single, rff_feature_map,\
    rff_feature_map_grad_d, rff_feature_map_grad2_d, rff_feature_map_grad,\
    rff_feature_map_grad2, rff_feature_map_grad_single
import numpy as np


def test_gaussian_kernel_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    gaussian_kernel_theano(x, y, sigma)

def test_gaussian_kernel_theano_result_equals_manual():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    k = gaussian_kernel_theano(x, y, sigma)
    k_manual = gaussian_kernel(x[np.newaxis, :], y[np.newaxis, :], sigma)[0, 0]
    
    assert_almost_equal(k, k_manual)

def test_gaussian_kernel_grad_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    gaussian_kernel_grad_theano(x, y, sigma)

def test_gaussian_kernel_grad_theano_result_equals_manual():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    grad = gaussian_kernel_grad_theano(x, y, sigma)
    grad_manual = gaussian_kernel_grad(x, y[np.newaxis, :], sigma)[0]
    print grad_manual
    print grad
    
    assert_allclose(grad, grad_manual)

def test_gaussian_kernel_hessian_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    gaussian_kernel_hessian_theano(x, y, sigma)

def test_gaussian_kernel_third_order_derivative_tensor_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    gaussian_kernel_third_order_derivative_tensor_theano(x, y, sigma)

def test_feature_map():
    x = 3.
    u = 2.
    omega = 2.
    phi = rff_feature_map_single(x, omega, u)
    phi_manual = np.cos(omega * x + u) * np.sqrt(2.)
    assert_close(phi, phi_manual)

def test_feature_map_single_equals_feature_map():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    phis = rff_feature_map(X, omega, u)
    
    for i, x in enumerate(X):
        phi = rff_feature_map_single(x, omega, u)
        assert_allclose(phis[i], phi)

def test_feature_map_derivative_d_1n():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative = rff_feature_map_grad_d(X, omega, u, d)
    phi_derivative_manual = -np.sin(X * omega + u) * omega[:, d] * np.sqrt(2.)
    assert_close(phi_derivative, phi_derivative_manual)

def test_feature_map_derivative_d_2n():
    X = np.array([[1.], [3.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative = rff_feature_map_grad_d(X, omega, u, d)
    phi_derivative_manual = -np.sin(X * omega + u) * omega[:, d] * np.sqrt(2.)
    assert_close(phi_derivative, phi_derivative_manual)

def test_feature_map_derivative2_d():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative2 = rff_feature_map_grad2_d(X, omega, u, d)
    phi_derivative2_manual = -rff_feature_map(X, omega, u) * (omega[:, d] ** 2)
    assert_close(phi_derivative2, phi_derivative2_manual)

def test_feature_map_derivatives_loop_equals_map_derivative_d():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = rff_feature_map_grad_loop(X, omega, u)
    
    for d in range(D):
        derivative = rff_feature_map_grad_d(X, omega, u, d)
        assert_allclose(derivatives[d], derivative)

def test_feature_map_derivatives_equals_feature_map_derivatives_loop():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = rff_feature_map_grad(X, omega, u)
    derivatives_loop = rff_feature_map_grad_loop(X, omega, u)
    
    assert_allclose(derivatives_loop, derivatives)

def test_feature_map_derivatives2_loop_equals_map_derivative2_d():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = rff_feature_map_grad2_loop(X, omega, u)
    
    for d in range(D):
        derivative = rff_feature_map_grad2_d(X, omega, u, d)
        assert_allclose(derivatives[d], derivative)

def test_feature_map_derivatives2_equals_feature_map_derivatives2_loop():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = rff_feature_map_grad2(X, omega, u)
    derivatives_loop = rff_feature_map_grad2_loop(X, omega, u)
    
    assert_allclose(derivatives_loop, derivatives)

def test_feature_map_grad_single_equals_feature_map_derivative_d():
    D = 2
    m = 3
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    x = np.random.randn(D)
    
    grad = rff_feature_map_grad_single(x, omega, u)
    
    grad_manual = np.zeros((D, m))
    for d in range(D):
        grad_manual[d, :] = rff_feature_map_grad_d(x, omega, u, d)
    
    assert_allclose(grad_manual, grad)
