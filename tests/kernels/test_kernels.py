from nose import SkipTest
from nose.tools import assert_almost_equal
from numpy.testing.utils import assert_allclose

from kernel_exp_family.kernels.kernels import gaussian_kernel_theano, \
    gaussian_kernel_grad_theano, gaussian_kernel_hessian_theano, \
    theano_available, gaussian_kernel_third_order_derivative_tensor_theano, gaussian_kernel, \
    gaussian_kernel_grad
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
